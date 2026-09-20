"""Persistent, single-GPU offload rollout worker. Never archives raw latents."""
import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import torch

from core import OFFICIAL, GENERATION, PROTOCOL, validate_checkpoint, validate_episode
from artifacts import (complete, directory, dump, gpu_info, identity, read, save, seal, sha,
                       space, verify_sources)
from runtime_reinforce import Policy, apply_policy


def write_video(video, path):
    import imageio.v2 as imageio
    import numpy as np
    rgb = video.detach().float().clamp(-1, 1).add(1).mul(.5).permute(1, 2, 3, 0).cpu().numpy()
    if rgb.shape != (81, 480, 832, 3) or not np.isfinite(rgb).all():
        raise ValueError('invalid video shape/values')
    with imageio.get_writer(str(path), fps=16, codec='libx264', quality=8,
            macro_block_size=None, ffmpeg_params=['-threads', '1']) as writer:
        for frame in rgb:
            writer.append_data((frame * 255).round().astype(np.uint8))


def performance(timing, profile, k):
    from reporting import extract_component_latency, extract_component_tflops
    if (timing['status'] != 'success' or timing['model_forward_call_count'] != 100
            or timing['full_compute_forward_calls'] != 2 * (50 - k)
            or timing['reuse_forward_calls'] != 2 * k
            or any(r['blocks_executed'] not in (0, 30) for r in timing['calls'])):
        raise ValueError('actual DiT block calls differ from Exact-K')
    f = profile['per_model_forward']
    seconds = timing['pipeline_generate_wall_seconds']
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('invalid full generation latency')
    return dict(generate_seconds=seconds,
        dit_tflops=sum(f['estimated_full_flops'] if c['blocks_executed'] == 30
                      else f['estimated_always_on_flops'] for c in timing['calls']) / 1e12,
        **extract_component_latency(timing), **extract_component_tflops(profile))


class Worker:
    def __init__(self, run, rank):
        self.run, self.rank = Path(run), rank
        self.config = read(self.run / 'config.json')
        verify_sources(self.config)
        self.profile = read(self.config['profile'])
        if sha(self.config['profile']) != self.config['profile_sha256']:
            raise ValueError('FLOPs profile changed')
        if Path(sys.prefix).name != 'wan2.2' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('requires wan2.2 environment and exactly one visible GPU')
        torch.set_num_threads(1); torch.cuda.set_device(0)
        prop = torch.cuda.get_device_properties(0)
        uuid = str(getattr(prop, 'uuid', '')) or gpu_info([os.environ['CUDA_VISIBLE_DEVICES']])[0]['uuid']
        self.gpu = self.config['gpus'][rank]
        if uuid != self.gpu['uuid']:
            raise ValueError('worker physical GPU changed')
        lock_path = Path('/tmp') / f'ours21_reinforce_{uuid}.lock'
        self.lock = lock_path.open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if torch.cuda.mem_get_info()[0] < 21 * 2**30:
            raise ValueError('GPU not idle enough for Wan21 offload; existing jobs are not interrupted')
        sys.path.insert(0, str(OFFICIAL / 'Wan21Benchmark'))
        from protocol import source_lock
        source_lock(Path(self.config['wan_root']))
        sys.path.insert(0, str(OFFICIAL / 'SeaCache4Wan21'))
        sys.path.insert(0, str(OFFICIAL / 'ComponentMetrics'))
        sys.path.insert(0, str(OFFICIAL / 'VideoMetrics'))
        sys.path.insert(0, self.config['wan_root'])
        import wan
        from wan.configs import WAN_CONFIGS
        with (self.run / 'initialization.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            began = time.perf_counter()
            self.pipe = wan.WanT2V(config=WAN_CONFIGS['t2v-1.3B'],
                checkpoint_dir=self.config['wan_weights'], device_id=0, rank=0,
                t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False)
            torch.cuda.synchronize()
            self.init_seconds = time.perf_counter() - began
        if self.pipe.param_dtype != torch.bfloat16 or len(self.pipe.model.blocks) != 30:
            raise ValueError('Wan architecture/dtype mismatch')
        self.native_forward = self.pipe.model.forward
        self.native_generate = self.pipe.generate
        self.policy = None
        self.warmed = False

    def native(self):
        self.pipe.model.forward = self.native_forward
        self.pipe.generate = self.native_generate

    def get_policy(self, job):
        if sha(job['checkpoint']) != job['checkpoint_sha256']:
            raise ValueError('checkpoint changed during rollout')
        payload = torch.load(job['checkpoint'], map_location='cpu', weights_only=False)
        validate_checkpoint(payload)
        if payload['run_id'] != self.config['run_id'] or payload['version'] != job['policy_version']:
            raise ValueError('foreign policy in job')
        if self.policy is None or self.policy.version != payload['version']:
            self.policy = Policy(payload, device='cuda:0', allow_smoke=self.config['smoke_only'])
        self.policy.set_sampling(job.get('sampling_seed'))
        return self.policy, payload

    def warmup(self, job):
        self.native()
        with torch.no_grad():
            native = self.pipe.generate(job['prompt'], **GENERATION)
        if self.config['smoke_only']:
            policy, _ = self.get_policy(job)
            native = native.cpu()
            apply_policy(self.pipe, policy, 0)
            with torch.no_grad():
                instrumented = self.pipe.generate(job['prompt'], **GENERATION).cpu()
            if not torch.equal(native, instrumented):
                raise ValueError('native vs Exact-K K=0 is not bitwise equivalent')
            del instrumented
            dump(self.run / f'native_equivalence_{self.rank}.json', dict(
                status='passed', equality='bitwise float32 VAE output', gpu=self.gpu,
                excluded_from_training=True, raw_latents_saved=False))
        del native
        self.native(); gc.collect(); torch.cuda.empty_cache()
        self.warmed = True

    def generate_measured(self, prompt, timing_path):
        from inference_timing import _PipelineProfiler
        profiler = _PipelineProfiler(self.pipe, init_wall_seconds=self.init_seconds,
                                      output_path=timing_path, implementation='reinforce_g1_offload')
        profiler.install()
        with torch.no_grad():
            return self.pipe.generate(prompt, **GENERATION)

    def job(self, job):
        expected = identity(self.config, job)
        output = self.run / job['output']
        temporary = directory(self.run / '_temporary', '# Ephemeral encoded videos; no raw latents. Safe to remove after metrics.')
        candidate_path = temporary / (job['id'] + '.mp4')
        native_path = temporary / (job['id'] + '.reference.mp4')
        if complete(output, expected):
            # Evaluation videos are staged until the separate VBench phase.
            if not job['evaluation'] or candidate_path.exists():
                return
            raise ValueError('evaluation video missing before VBench completion; start a new evaluation directory')
        if output.exists() and any(output.iterdir()):
            archive = directory(self.run / '_incomplete', '# Incomplete small artifacts; never used for updates.')
            output.rename(archive / (job['id'] + f'_{time.time_ns()}'))
        directory(output, '# Pooled features, trace, reward and timings only; no raw latents or retained training video.')
        space(self.run, len(self.config['gpus']))
        if not self.warmed:
            self.warmup(job)
        self.native()
        base = job.get('baseline')
        baseline_measurement = None
        try:
            # Evaluation always measures an on-card native reference for fair latency.
            if base and not job['evaluation']:
                if base['gpu_uuid'] != self.gpu['uuid'] or sha(base['path']) != base['sha256']:
                    raise ValueError('reference video/GPU changed')
                reference_path = Path(base['path'])
            else:
                reference_path = native_path
                video = self.generate_measured(job['prompt'], output / 'baseline_timing.json')
                baseline_measurement = performance(read(output / 'baseline_timing.json'), self.profile, 0)
                write_video(video, reference_path); del video
            policy, checkpoint = self.get_policy(job)
            ctrl = apply_policy(self.pipe, policy, job['k'])
            video = self.generate_measured(job['prompt'], output / 'timing.json')
            trace = ctrl.summary()
            timing = read(output / 'timing.json')
            timing['predictor'] = policy.overhead_summary()
            timing['latent_feature'] = ctrl.feature_overhead()
            dump(output / 'timing.json', timing)
            perf = performance(timing, self.profile, job['k'])
            from ours4wan21.overhead import predictor_fields
            perf.update(predictor_fields(timing, trace))
            perf['feature_wall_seconds'] = ctrl.feature_overhead()['wall_seconds']
            perf['feature_tflops'] = None
            write_video(video, candidate_path); del video
            from video_metrics.evaluator import evaluate_pairs, VideoPair
            metrics = ('psnr', 'ssim', 'lpips') if job['evaluation'] else ('psnr',)
            frame_rows, video_rows, _ = evaluate_pairs(
                [VideoPair(job['id'], reference_path, candidate_path)], metrics=metrics,
                device='cuda', lpips_batch_size=1, expected_frames=81)
            reward = float(video_rows[0]['psnr_rgb_db_mean'])
            episode = ctrl.episode(job, checkpoint, reward)
            if not job['evaluation']:
                validate_episode(episode, checkpoint)
            save(output / 'episode.pt', episode)
            dump(output / 'trace.json', trace)
            dump(output / 'quality.json', dict(video=video_rows[0], frames=frame_rows,
                 scope='VideoMetrics decoded RGB; mean per-frame PSNR; terminal reward',
                 temporary_video_deleted_after_metrics=not job['evaluation']))
            if baseline_measurement:
                perf['native_baseline'] = baseline_measurement
                perf['speedup'] = baseline_measurement['generate_seconds'] / perf['generate_seconds']
            else:
                perf['speedup'] = None
                perf['speedup_scope'] = 'existing diagnostic baseline used for PSNR only; no speed ratio inferred'
            dump(output / 'measurement.json', perf)
            dump(output / 'identity.json', expected)
            names = ['episode.pt', 'trace.json', 'quality.json', 'measurement.json', 'timing.json', 'identity.json']
            if baseline_measurement:
                names.append('baseline_timing.json')
            seal(output, expected, names)
            print(json.dumps(dict(completed=job['id'], reward=reward, k=job['k'])), flush=True)
        finally:
            if not job['evaluation']:
                native_path.unlink(missing_ok=True)
                candidate_path.unlink(missing_ok=True)
            self.native()
            # Release branch residuals and GPU feature history before the next T5 encode.
            if hasattr(self.pipe.model, 'seacache_controller'):
                del self.pipe.model.seacache_controller
            gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    worker = None
    for line in sys.stdin:
        message = json.loads(line)
        if message.get('stop'):
            break
        jobs = read(message['jobs'])
        if jobs and worker is None:
            worker = Worker(args.run, args.rank)
        for job in jobs:
            if job['worker'] != args.rank:
                raise ValueError('misrouted GPU job')
            worker.job(job)
        dump(message['ack'], dict(status='complete', jobs=[j['id'] for j in jobs],
                                  request_sha256=sha(message['jobs'])))


if __name__ == '__main__':
    main()
