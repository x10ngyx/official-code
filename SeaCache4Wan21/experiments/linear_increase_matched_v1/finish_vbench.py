"""Resume missing official VBench custom dimensions without recomputing complete scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import run_experiment as s


def evaluate(label):
    sys.path.insert(0,str(s.OFFICIAL/'VbenchEvaluation'))
    from evaluate_custom_vbench import CUSTOM_DIMENSIONS,load_prompt_map
    import torch
    from vbench import VBench
    cfg=s.read(s.ROOT/'config.json');d=s.ROOT/'conditions'/label;videos=d/'videos';scores=d/'vbench/scores'
    s.mkdir(d,'# Condition\n\nFour paired GPU trajectories and official VBench custom dimension results.')
    videos.mkdir(exist_ok=True)
    for g in range(4):
        sid=cfg['prompts'][str(g)]['sample_id'];link=videos/f'{sid}.mp4'
        if not link.is_symlink():link.symlink_to(d/f'gpu{g}'/'videos'/f'{sid}.mp4')
    s.dump(d/'prompt_map.json',{f"{r['sample_id']}.mp4":r['prompt_en'] for r in cfg['prompts'].values()})
    prompts=load_prompt_map(d/'prompt_map.json',videos);scores.mkdir(parents=True,exist_ok=True)
    completed=[];remaining=[]
    for dimension in CUSTOM_DIMENSIONS:
        file=scores/f'vbench_custom_{dimension}_eval_results.json'
        if file.exists():
            value=s.read(file)[dimension]
            if not math.isfinite(float(value[0])) or len(value[1])!=4 or {Path(x['video_path']).name for x in value[1]}!=set(prompts):raise ValueError('existing dimension coverage invalid')
            completed.append(dimension)
        else:remaining.append(dimension)
    s.dump(d/'vbench/RESUME_MANIFEST.json',dict(official_full_vbench_score=False,completed_dimensions=completed,remaining_dimensions=remaining,video_sha256={str(p):s.sha(p) for p in videos.glob('*.mp4')},prompt_map_sha256=s.sha(d/'prompt_map.json'),script_sha256=s.sha(Path(__file__)),tokenizer_sha256=s.sha(s.pair.prior.MODEL_ROOT/'VBench/ViCLIP/bpe_simple_vocab_16e6.txt.gz')))
    evaluator=VBench(torch.device('cuda'),str(s.OFFICIAL/'Vbench200/VBench200_full_info.json'),str(scores))
    for dimension in remaining:
        evaluator.evaluate(videos_path=str(videos),name=f'vbench_custom_{dimension}',prompt_list=prompts,dimension_list=[dimension],local=True,mode='custom_input',imaging_quality_preprocessing_mode='longer')
    subprocess.run([sys.executable,str(s.OFFICIAL/'VbenchEvaluation/aggregate_custom_vbench_scores.py'),'--score-dir',str(scores),'--output',str(d/'vbench/vbench_custom_aggregate_scores.json')],check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition');a=p.parse_args()
    if a.condition:evaluate(a.condition);return
    cfg=s.prepare();root=s.ROOT
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHON_BIN=sys.executable,TORCH_HOME=str(s.pair.prior.MODEL_ROOT/'torch-cache'),VBENCH_CACHE_DIR=str(s.pair.prior.MODEL_ROOT/'VBench'),HF_HOME=str(s.pair.prior.MODEL_ROOT/'VBench/huggingface'),XDG_CACHE_HOME=str(s.pair.prior.MODEL_ROOT/'VBench/xdg'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (root/'queue.lock').open('a') as local,(s.pair.EXP/'wan21_benchmark_4gpu.lock').open('a') as shared:
        fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(shared,fcntl.LOCK_EX)
        failed=root/'FAILED.json'
        if failed.exists():failed.rename(root/'recovery/dino_cache_alias/tokenizer_race_failure.json')
        s.dump(root/'status.json',dict(status='vbench_custom_resume_missing_dimensions',generation_complete=40,quality_complete=40,updated_at=s.pair.now()))
        def queue(g):
            for i in range(5):
                if i%4!=g:continue
                for phase in ('increase','fixed'):
                    label=f'{phase}_{i+1}';d=root/'conditions'/label
                    if (d/'vbench/vbench_custom_aggregate_scores.json').exists():continue
                    with (root/'logs'/f'vbench_finish_{label}.log').open('ab') as log:
                        subprocess.run([sys.executable,str(Path(__file__)),'--condition',label],cwd=s.OFFICIAL,env=dict(env,CUDA_VISIBLE_DEVICES=cfg['gpu_uuids'][str(g)]),stdout=log,stderr=subprocess.STDOUT,check=True)
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(queue,range(4)))
            s.finalize(cfg)
            import audit
            audit.main()
            s.dump(root/'status.json',dict(status='complete',updated_at=s.pair.now()))
        except BaseException as exc:
            s.dump(root/'FAILED.json',dict(error=repr(exc),at=s.pair.now()));raise
if __name__=='__main__':main()
