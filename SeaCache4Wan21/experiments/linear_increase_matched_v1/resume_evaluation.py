"""Resume only evaluation after all 40 generations; preserves frozen generation code."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import run_experiment as s


def main():
    cfg=s.prepare();root=s.ROOT
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHON_BIN=sys.executable,TORCH_HOME=str(s.pair.prior.MODEL_ROOT/'torch-cache'),VBENCH_CACHE_DIR=str(s.pair.prior.MODEL_ROOT/'VBench'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (root/'queue.lock').open('a') as local,(s.pair.EXP/'wan21_benchmark_4gpu.lock').open('a') as shared:
        fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(shared,fcntl.LOCK_EX)
        for i in range(5):
            for phase in ('increase','fixed'):
                for g in range(4):
                    row,_=s.condition(cfg,phase,i,g)
                    s.pair.prior.quality_rows(root/'conditions'/f'{phase}_{i+1}'/f'gpu{g}'/'quality',[row['sample_id']])
        s.dump(root/'status.json',dict(status='vbench_custom',generation_complete=40,quality_complete=40,updated_at=s.pair.now()))
        def queue(g):
            for i in range(5):
                if i%4!=g:continue
                for phase in ('increase','fixed'):
                    d=root/'conditions'/f'{phase}_{i+1}'
                    s.mkdir(d,'# Condition\n\nFour GPU trajectories, videos/ symlink view and VBench custom metrics.')
                    (d/'videos').mkdir(exist_ok=True)
                    for gpu in range(4):
                        sid=cfg['prompts'][str(gpu)]['sample_id'];link=d/'videos'/f'{sid}.mp4'
                        if not link.is_symlink():link.symlink_to(d/f'gpu{gpu}'/'videos'/f'{sid}.mp4')
                    s.dump(d/'prompt_map.json',{f"{r['sample_id']}.mp4":r['prompt_en'] for r in cfg['prompts'].values()})
                    if (d/'vbench/vbench_custom_aggregate_scores.json').exists():continue
                    cmd=['bash',str(s.OFFICIAL/'VbenchEvaluation/run_custom_vbench.sh'),str(d/'videos'),str(d/'vbench'),str(d/'prompt_map.json')]
                    with (root/'logs'/f'vbench_resume_{phase}_{i+1}.log').open('ab') as log:
                        subprocess.run(cmd,cwd=s.OFFICIAL,env=dict(env,CUDA_VISIBLE_DEVICES=cfg['gpu_uuids'][str(g)]),stdout=log,stderr=subprocess.STDOUT,check=True)
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(queue,range(4)))
            s.finalize(cfg)
            import audit
            audit.main()
            s.dump(root/'status.json',dict(status='complete',updated_at=s.pair.now()))
        except BaseException as e:
            s.dump(root/'FAILED.json',dict(error=repr(e),at=s.pair.now()))
            s.dump(root/'status.json',dict(status='evaluation_failed',generation_complete=40,quality_complete=40,error=repr(e)))
            raise
if __name__=='__main__':main()
