#!/usr/bin/env python3
"""Build a small self-contained code/dependency handoff; no weights or results."""
import argparse
from pathlib import Path
import tarfile
from common import PACKAGES, PROJECT, external, sha, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    output=external(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    archive=output/'cache_impact_wan21_remote.tar.gz'
    if archive.exists():raise FileExistsError(archive)
    files=set()
    for name in ['CacheImpact4Wan21','ComponentMetrics','VideoMetrics','VbenchEvaluation',
                 'Vbench200','CalflopsEvaluation']:
        for f in (PACKAGES/name).rglob('*'):
            if not f.is_file() or f.is_symlink():continue
            relative=f.relative_to(PACKAGES/name)
            if any(x in relative.parts for x in ['experiment_results','__pycache__','.git','logs','.pytest_cache']):continue
            if name!='CacheImpact4Wan21' and any(x in relative.parts for x in ['experiments','tests']):continue
            if f.suffix not in {'.py','.sh','.md','.json','.jsonl','.txt','.toml','.yaml','.yml'}:continue
            files.add(f)
    for f in (PACKAGES/'SeaCache4Wan21').iterdir():
        if f.is_file() and f.suffix in {'.py','.sh','.md','.json'}:files.add(f)
    for name in ['Wan21Benchmark/protocol.py','Wan21Benchmark/experiments/component_profile/profile_calflops.py',
                 'Wan21Benchmark/experiments/component_profile/README.md','DiCache4Wan21/upstream_lock.json']:
        files.add(PACKAGES/name)
    manifest={str(Path('work/offical-code')/f.relative_to(PACKAGES)):sha(f) for f in sorted(files)}
    with tarfile.open(archive,'w:gz') as tar:
        for f in sorted(files):tar.add(f,arcname=str(Path('work/offical-code')/f.relative_to(PACKAGES)),recursive=False)
    write_json(output/'files.json',manifest)
    write_json(output/'bundle.json',dict(archive=archive.name,sha256=sha(archive),files=len(files),
               excluded='model weights, Wan source checkout, prompts40, all experiment outputs'))
    (output/'README.md').write_text('# Remote code bundle\n\nExtract archive at the workspace root: tar -xzf cache_impact_wan21_remote.tar.gz -C /path/to/workspace. It contains work/offical-code/CacheImpact4Wan21 and its shared code dependencies, not models or the external locked Wan checkout. Follow CacheImpact4Wan21/experiments/single_skip_v1/README.md. files.json has source hashes; bundle.json has the archive SHA.\n')
    link=PROJECT/'experiment_results'/output.name
    if not link.exists():link.symlink_to(output,target_is_directory=True)
    print(archive)


if __name__=='__main__':main()
