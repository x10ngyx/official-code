"""Validate all newly imported packages without loading models."""
import hashlib
import json
from pathlib import Path

PACKAGES=Path(__file__).resolve().parent.parent


def main():
    report={}
    for name in ['MagCache4Wan21','DiCache4Wan21','TaylorSeer4Wan21']:
        root=PACKAGES/name
        lock=json.loads((root/'upstream_lock.json').read_text())
        for file,sha in lock['integration_artifact_sha256'].items():
            if hashlib.sha256((root/file).read_bytes()).hexdigest()!=sha:
                raise ValueError(f'source hash mismatch: {name}/{file}')
        report[name]=dict(source_files=len(lock['integration_artifact_sha256']),status='pass')
    for name in [*report,'Wan21Benchmark']:
        for file in (PACKAGES/name).rglob('*.py'):
            if 'experiment_results' not in file.parts:
                compile(file.read_bytes(),str(file),'exec')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
