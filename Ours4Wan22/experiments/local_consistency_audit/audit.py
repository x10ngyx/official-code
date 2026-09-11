"""Read-only source and CPU numerical comparisons; no Wan model loading."""
import argparse
import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key) == '1', key
assert Path(sys.prefix).name == 'wan2.2'
import torch

torch.set_num_threads(1)
PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[2]
sys.path.insert(0, str(PROJECT))
from ours4wan22.shared import result_dir, sha256, write
from ours4wan22.runtime import sea


def functions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    local = ROOT / 'work/Wan2.2/predicotr-rl'
    frozen = PROJECT.parent / 'Ours4Wan21/ours4wan21/local_iql.py'
    comparisons = {}
    files = [frozen, local/'train_iql_hard_budget.py', local/'hard_budget.py']
    new = functions(frozen)
    for source, names in ((files[1], ('_move_batch', '_actor_terms', '_run_epoch')),
                          (files[2], ('normalize_forced_steps', 'max_skip_budget',
                                      'target_skip_budget_from_calibration', 'required_hard_budget_action'))):
        old = functions(source)
        for name in names:
            comparisons[name] = new[name] == old[name]
    assert all(comparisons.values()), comparisons
    path = ROOT/'work/Wan2.2/wan/timestep_cache.py'
    spec = importlib.util.spec_from_file_location('audit_local_cache', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    files += [path, PROJECT.parent/'SeaCache4Wan22/runtime/seacache.py']
    torch.manual_seed(42)
    old_cache = module.SeaCacheTimestepCache(module.SeaCacheTimestepCacheConfig())
    new_cache = sea.SeaCacheController(sea.SeaCacheConfig(threshold=1.))
    sigmas = torch.linspace(1, 0, 51)
    new_cache.set_scheduler_sigmas(sigmas)
    max_error = 0.
    cases = 0
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        previous = None
        for step in range(50):
            feature = torch.randn(1, 24, 8).to(dtype)
            grid = torch.tensor([2, 3, 4])
            old = old_cache._filter_feature(feature, grid, step, 50, sigmas)
            new_value = new_cache._filter_feature(feature, grid, step, 50)
            max_error = max(max_error, float((old.float()-new_value.float()).abs().max()))
            assert torch.equal(old, new_value)
            if previous is not None:
                assert old_cache._relative_l1(old, previous) == new_cache._relative_l1(new_value, previous)
            previous = old
            cases += 1
    source = Path('/all/yiran07-disk3/huteng_data/exp/wan22_pooled1287_sea_distance_20260905/random2323/dataset/data/tables/trajectory_summary.csv')
    with source.open() as f:
        row = next(csv.DictReader(f))
    provenance = {k: row.get(k) for k in ('trajectory_id', 'split', 'mean_psnr', 'candidate_video',
        'baseline_video', 'inference_compute_elapsed_seconds', 'baseline_inference_compute_elapsed_seconds',
        'candidate_timing_normalization_applied', 'baseline_timing_normalization_applied')}
    psnr_path = Path(row['candidate_video']).parents[1]/'psnr'/(Path(row['candidate_video']).stem+'.json')
    psnr = json.loads(psnr_path.read_text())
    provenance['original_psnr_method'] = psnr['method']
    provenance['original_psnr_matches_training_table'] = float(row['mean_psnr']) == psnr['mean_psnr']
    assert provenance['original_psnr_matches_training_table']
    files.append(psnr_path)
    files.append(source)
    output = result_dir(args.output, '# Local consistency audit\n\nvalidation.json: AST parity, CPU filter comparisons, dataset provenance and source hashes. See project docs/local_consistency_audit_20260911.md for scope and findings. No GPU experiment.')
    write(output/'validation.json', dict(status='pass', ast_equal=comparisons,
        sea_filter_cases=cases, sea_filter_max_abs_error=max_error,
        exact_tie_actions=dict(local_threshold_ge_half=1, new_argmax=0),
        dataset_sample=provenance, source_sha256={str(p):sha256(p) for p in files},
        limitation='Small CPU tensors and AST comparisons; no full 14B numerical or performance validation.'))
    print((output/'validation.json').read_text())


if __name__ == '__main__':
    main()
