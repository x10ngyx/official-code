"""Local suite context; reuse established evidence validators."""
import importlib.util
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('established_iql_common',PROJECT/'experiments/iql_aggressiveness_2x4_v1/common.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
for name in ('EXP_ROOT','MODEL_ROOT','OFFICIAL','PROTOCOL','WORKSPACE','training_config','dump','sha256',
             'load_evaluation_bundle','gpu_uuids','TARGETS','BUDGETS','FIELDS','METRICS','read','writecsv',
             'environment','verify_training','verify_sources','audit_trace','REFERENCE'):
    globals()[name]=getattr(legacy,name)
ROOT=EXP_ROOT/'ours21_increase500_iql2_v1'
COLLECTION=EXP_ROOT/'ours21_increase500_v1'
FEATURE_SOURCE=EXP_ROOT/'ours21_random3000_12groups_v1_features'
MODE='sea7_dynamics_raw_sea128'
