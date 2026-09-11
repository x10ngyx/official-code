"""Read-only validation of the finished paired50 nominal3.6x experiment."""
import json
import math
from pathlib import Path
import statistics
import sys

OFFICIAL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OFFICIAL / 'Ours4Wan21/experiments/vbench50_speed36_pair_v1'))
import run_pair as pair


def audit():
    cfg = pair.prepare(True)
    root = pair.ROOT
    done = pair.read(root / 'COMPLETE.json')
    assert done['status'] == 'complete' and done['candidates'] == 100
    for filename, key in [('results.csv','results_sha256'), ('VALIDATION.json','validation_sha256')]:
        assert pair.sha(root / 'analysis' / filename) == done[key]
    validation = pair.read(root / 'analysis/VALIDATION.json')
    assert pair.sha(root / 'config.json') == validation['config_sha256']
    assert pair.sha(root / 'calibration/CALIBRATED.json') == validation['calibration_sha256']
    for name, digest in validation['evidence_sha256'].items():
        assert pair.sha(root / name) == digest
    for name, digest in cfg['baseline_sha256'].items():
        assert pair.sha(Path(name)) == digest
    summaries = pair.read_csv(root / 'analysis/results.csv')
    details = pair.read_csv(root / 'analysis/per_video.csv')
    assert len(details) == len({(r['method'],r['sample_id']) for r in details}) == 100
    for summary in summaries:
        method = summary['method']; base = []; candidate = []; quality = []
        for gpu in pair.prior.GPUS:
            ids = cfg['shard_ids'][str(gpu)]
            directory = root / 'shards' / f'gpu{gpu}' / method
            base += pair.read(Path(cfg['previous']) / 'shards' / f'gpu{gpu}' / 'baseline/components.json')['rows']
            candidate += pair.check_generation(directory, gpu, method, ids, cfg)
            quality += pair.prior.quality_rows(directory / 'quality', ids)
        candidate = [dict(r, **{k:0. for k in pair.prior.FIELDS[7:] if k not in r}) for r in candidate]
        for key, value in pair.prior.summarize(base, candidate).items():
            assert math.isclose(value, float(summary[key]), abs_tol=1e-9)
        selected = [r for r in details if r['method'] == method]
        assert len(selected) == 50
        for metric in pair.prior.METRICS:
            assert math.isclose(statistics.fmean(float(r[metric+'_mean']) for r in quality), float(summary[metric]), abs_tol=1e-10)
            assert math.isclose(statistics.fmean(float(r[metric]) for r in selected), float(summary[metric]), abs_tol=1e-10)
    return dict(status='pass', completed_at=done['completed_at'], formal_candidates=100, quality_frames=8100,
                evidence_files=len(validation['evidence_sha256']), source_files=len(cfg['source_sha256']), results=summaries)


if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
