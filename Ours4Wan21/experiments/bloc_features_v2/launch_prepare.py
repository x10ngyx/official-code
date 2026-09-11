"""Four disjoint extraction workers followed by verified training-cache assembly."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    exp=Path('/mnt/hdd/xiongyuxiang/tmp/exp')
    common=[sys.executable,str(HERE/'prepare.py'),
            '--selection',str(exp/'ours21_random3000_selection_v1/selection.json'),
            '--collection-root',str(exp/'wan21_random_threshold_collection_v1_stage1'),
            '--sea7-cache',str(exp/'ours21_random3000_12groups_v1_sea7_cache'),
            '--output-dir',str(a.output_dir),'--workers','4']
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             NUMEXPR_NUM_THREADS='1',CUDA_DEVICE_ORDER='PCI_BUS_ID')
    subprocess.run(common+['--phase','initialize'],env=env,check=True)
    def worker(gpu):
        with (a.output_dir/'workers'/f'{gpu}.log').open('w') as log:
            subprocess.run(common+['--phase','extract','--worker',str(gpu)],
                           env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker,range(4)))
    subprocess.run(common+['--phase','finalize'],env=env,check=True)


if __name__=='__main__':
    main()
