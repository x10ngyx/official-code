import importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT))
p=PROJECT/'experiments/iql_aggressiveness_2x4_v1/common.py'
s=importlib.util.spec_from_file_location('cnn_eval_legacy_common',p);legacy=importlib.util.module_from_spec(s);s.loader.exec_module(legacy)
for name in ('EXP_ROOT','MODEL_ROOT','OFFICIAL','PROTOCOL','WORKSPACE','REFERENCE','BUDGETS','TARGETS','FIELDS','METRICS','read','writecsv','sha256','dump','load_evaluation_bundle','gpu_uuids','environment','verify_sources','audit_trace'):
    globals()[name]=getattr(legacy,name)
ROOT=EXP_ROOT/'ours21_cnn_vbench20_v1'
TRAIN=EXP_ROOT/'ours21_cnn_mixed3500_v1'
FORMAL=PROJECT/'experiments/cnn_mixed3500_v1'
