"""Reuse native baselines from the exact offline training prompt population."""
import json
import math
from pathlib import Path
from fractions import Fraction
from .contracts import PROTOCOL, PROJECT, create_result, dump, sha256
from .online_common import prompts, read, seal, verified
from .online_reference import gpu_uuids

COLLECTION_PROTOCOL=dict(model='Wan2.1-T2V-1.3B',task='t2v-1.3B',size_wh=[832,480],
    frame_num=81,fps=16,sample_steps=50,sample_solver='unipc',sample_shift=5.,cfg=5.,
    seed=42,parameter_dtype='bfloat16',offload_model=False,t5_cpu=False)


def offline_train_pool(dataset_manifest, registry):
    ids={sid for sid,split in dataset_manifest['prompt_splits'].items() if split=='train'}
    rows=sorted((r for r in registry if r['sample_id'] in ids),key=lambda r:r['sample_id'])
    if not ids or {r['sample_id'] for r in rows}!=ids:
        raise ValueError('registry must cover the exact offline training population')
    return rows


def load_training_bundle(folder, *, verify_artifacts=False):
    folder=Path(folder)
    if not verified(folder):raise ValueError('incomplete training baseline reference')
    m=read(folder/'manifest.json')
    if m['schema']!='ours21_online_offline_train_reference_v1' or m['protocol']!=PROTOCOL:
        raise ValueError('wrong training baseline reference protocol')
    for path,digest in m['source_files'].items():
        if sha256(path)!=digest:raise ValueError('training baseline source changed: '+path)
    if len({r['sample_id'] for r in m['rows']})!=len(m['rows']):raise ValueError('duplicate training prompt')
    if verify_artifacts:
        for row in m['rows']:
            if not verified(folder/'baselines'/row['sample_id']):raise ValueError('training baseline changed')
    return m


def build_training_bundle(dataset_manifest, registry, source, output):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():return load_training_bundle(output,verify_artifacts=True)
    dataset_manifest=Path(dataset_manifest).resolve();registry=Path(registry).resolve()
    rows=offline_train_pool(read(dataset_manifest),prompts(registry))
    uuids=gpu_uuids(['0','1','2','3'])
    launcher=PROJECT/'data_collection/experiments/random_threshold_collection_v1/launch_4gpu.sh'
    provenance='GPU index reconstructed from archived shard_index and original launcher CUDA_VISIBLE_DEVICES=$gpu; UUID resolved on current same host. Original archive did not record UUID.'
    sources={str(p):sha256(p) for p in (dataset_manifest,registry,launcher)}
    configs={}
    for g in range(4):
        p=source/'shards'/f'shard_{g:02d}'/'baseline_config_prefix_001000.json'
        cfg=read(p);sources[str(p)]=sha256(p)
        if cfg['protocol']!=COLLECTION_PROTOCOL or cfg['shard_index']!=g or cfg['mode']!='baseline':
            raise ValueError('offline baseline shard protocol differs')
        if sha256(cfg['flops_profile'])!=cfg['flops_profile_sha256']:raise ValueError('profile changed')
        configs[g]=cfg
    create_result(output,'# Offline training baseline reference\n\ntrain_prompts.jsonl contains the exact offline train population. baselines/ holds sealed adapters with symlinks to native archived videos, timing and traces. manifest.json records source hashes and the explicit GPU provenance limitation. Validation/test excluded.\n')
    (output/'baselines').mkdir();(output/'baselines/README.md').write_text('Sealed native baseline references by training prompt ID; large files remain in original archive.\n')
    enriched=[]
    for row in rows:
        sid=row['sample_id'];base=source/'shared_baselines'/sid
        marker=read(base/'BASELINE_COMPLETE.json');trace=read(base/'trace.json')
        timing=read(base/'timing.json');perf=read(base/'performance.json');probe=read(base/'ffprobe.json')
        archived=trace['manifest_record'];g=archived['shard_index'];cfg=configs[g]
        if (marker['schema']!='ours4wan21_baseline_complete_v1' or marker['sample_id']!=sid or
            marker['split']!='train' or marker['protocol']!=COLLECTION_PROTOCOL or
            archived['sample_id']!=sid or archived['prompt'].strip()!=row['prompt'] or archived['split']!='train' or
            trace['reuse']!=0 or trace['recompute']!=50 or timing['status']!='success' or
            perf['full_compute_forward_calls']!=100 or perf['reuse_forward_calls']!=0 or
            perf['flops_profile_sha256']!=cfg['flops_profile_sha256'] or
            perf['pipeline_generate_wall_seconds']!=marker['pipeline_generate_wall_seconds'] or
            timing['pipeline_generate_wall_seconds']!=marker['pipeline_generate_wall_seconds']):
            raise ValueError('invalid archived native training baseline: '+sid)
        stream=probe['streams'][0]
        if (stream['width'],stream['height'],int(stream['nb_read_frames']))!=(832,480,81) or Fraction(stream['r_frame_rate'])!=16:
            raise ValueError('invalid archived baseline geometry')
        measurement=dict(perf,generate_seconds=perf['pipeline_generate_wall_seconds'],dit_tflops=perf['estimated_dit_tflops_per_video'])
        for key in ('generate_seconds','dit_tflops','t5_cuda_seconds','dit_cuda_seconds','vae_decode_cuda_seconds','estimated_t5_tflops_per_video','estimated_vae_decode_tflops_per_video'):
            if not math.isfinite(measurement[key]) or measurement[key]<=0:raise ValueError('invalid component measurement')
        folder=output/'baselines'/sid;folder.mkdir()
        identity=dict(job=dict(kind='baseline',**row),source_baseline=str(base),protocol=PROTOCOL)
        dump(folder/'identity.json',identity)
        dump(folder/'generation.json',dict(identity=identity,protocol=PROTOCOL,gpu_uuid=uuids[g],
            gpu_provenance=provenance,source_shard_index=g,wan_checkpoint=cfg['checkpoint_dir'],
            flops_profile_sha256=cfg['flops_profile_sha256']))
        dump(folder/'measurement.json',measurement)
        links={'video.mp4':'baseline.mp4','trace.json':'trace.json','timing.json':'timing.json',
               'ffprobe.json':'ffprobe.json','source_complete.json':'BASELINE_COMPLETE.json','source_performance.json':'performance.json'}
        for name,filename in links.items():(folder/name).symlink_to(base/filename)
        (folder/'README.md').write_text('Reused offline train native baseline. Original video/timing/trace remain symlinked. GPU provenance is reconstructed, see generation.json.\n')
        seal(folder,['identity.json','generation.json','measurement.json',*links],identity=identity)
        enriched.append(dict(**row,baseline_gpu_uuid=uuids[g]))
    m=dict(schema='ours21_online_offline_train_reference_v1',protocol=PROTOCOL,source_run=str(source),
        dataset_manifest_sha256=sha256(dataset_manifest),source_files=sources,rows=enriched,gpu_provenance=provenance)
    dump(output/'manifest.json',m)
    (output/'train_prompts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    seal(output,['manifest.json','train_prompts.jsonl',*['baselines/'+r['sample_id']+'/COMPLETE.json' for r in rows]],identity=dict(dataset=sha256(dataset_manifest)))
    return load_training_bundle(output)
