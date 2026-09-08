import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5]
PACKAGES=ROOT/'work/offical-code'
OUTPUT=Path('/all/yiran07-disk3/huteng_data/exp/wan21_method_integration_20260908')
old=json.loads((OUTPUT/'existing_files_before.json').read_text())
changed=[p for p,sha in old.items() if not (PACKAGES/p).is_file() or hashlib.sha256((PACKAGES/p).read_bytes()).hexdigest()!=sha]
assert changed==['README.md'],changed
commands=[['Wan21Benchmark/validate.py']]
commands += [[name+'/experiments/fixed_protocol/run.py','--help'] for name in ['MagCache4Wan21','DiCache4Wan21','TaylorSeer4Wan21']]
commands += [['TaylorSeer4Wan21/generate.py','--help'],['Wan21Benchmark/metrics.py','--help'],['Wan21Benchmark/experiments/component_profile/profile_calflops.py','--help']]
checks=[]
for args in commands:
    run=subprocess.run([sys.executable,*args],cwd=PACKAGES,text=True,capture_output=True,
                       env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    checks.append(dict(command=args,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
    assert run.returncode==0,checks[-1]
result=dict(status='pass',existing_source_files_unchanged=len(old)-1,
            allowed_index_update=changed,commands=checks,
            validation_scope='CPU only; no full model inference or quality model execution')
(OUTPUT/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='commands'},indent=2))
