"""Independently validate final aggregates and pair archived e391 on the same ten prompts."""
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

ROOT = Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_iql_aggressiveness_2x4_v1')
OLD = ROOT.parent / 'ours21_dynamics128_e391_vbench50_random42_4gpu_v1'
OUT = ROOT / 'final_readout'
METRICS = ['psnr_rgb_db', 'ssim_rgb', 'lpips_alex_v0_1_spatial']

def read(p): return json.loads(p.read_text())
def rows(p): return list(csv.DictReader(p.open()))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
def writecsv(p, rr):
    with p.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)

def main():
    config = read(ROOT / 'config.json')
    done = read(ROOT / 'COMPLETE.json'); validation = read(ROOT / 'VALIDATION.json')
    evidence = {}
    def seal(p, expected=None):
        h = sha(p)
        if expected is not None: assert h == expected, p
        evidence[str(p)] = h
    for f, k in [('results.csv','results_sha256'),('REPORT.md','report_sha256'),('VALIDATION.json','validation_sha256')]:
        seal(ROOT / f, done[k])
    for p, h in validation['evidence_sha256'].items(): seal(Path(p), h)
    detail = rows(ROOT / 'per_video.csv'); agg = rows(ROOT / 'results.csv')
    seal(ROOT / 'per_video.csv')
    assert len(detail) == len({(r['group'],r['k'],r['sample_id']) for r in detail}) == 240
    ids = {r['sample_id'] for r in config['prompts']}
    for a in agg:
        rr = [r for r in detail if (r['group'],r['k']) == (a['group'],a['k'])]
        assert len(rr) == 10 and {r['sample_id'] for r in rr} == ids
        for key in METRICS + ['generate_seconds','dit_tflops','late25','longest_late25']:
            assert abs(st.fmean(float(r[key]) for r in rr) - float(a[key])) < 1e-8
        assert abs(sum(float(r['baseline_seconds']) for r in rr) / sum(float(r['generate_seconds']) for r in rr) - float(a['latency_speedup'])) < 1e-10
        for r in rr:
            p = r['skip_path']; assert len(p) == 50 and p.count('1') == int(r['k'])
            assert p[25:].count('1') == int(r['late25'])
            assert max(map(len,p[25:].split('0'))) == int(r['longest_late25'])
    old = [r for r in rows(OLD / 'analysis/per_video.csv') if r['sample_id'] in ids]
    assert len(old) == 30
    oldval = read(OLD / 'analysis/QUALITY_VALIDATION.json')
    for p, h in oldval['evidence_sha256'].items(): seal(OLD / p, h)
    seal(OLD / 'analysis/per_video.csv')
    old_index = {}
    for r in old:
        sid, k, gpu = r['sample_id'], r['k'], r['gpu']
        matched = next(x for x in detail if (x['sample_id'],x['k']) == (sid,k))
        assert matched['gpu'] == gpu
        path = OLD / f'shards/gpu{gpu}/K{k}'
        q = next(x for x in rows(path / 'quality/per_video.csv') if x['video_id'] == sid)
        assert (int(q['frames']),int(q['width']),int(q['height'])) == (81,832,480)
        for side in ['reference','candidate']: seal(Path(q[side]), q[side+'_sha256'])
        seal(Path(config['reference']) / 'baselines' / sid / 'video.mp4',q['reference_sha256'])
        for metric in METRICS: assert abs(float(r[metric])-float(q[metric+'_mean'])) < 1e-10
        tracepath = path / 'traces' / (sid+'.json'); seal(tracepath)
        trace = read(tracepath)
        skips = trace['per_branch']['cond']['reuse_path']
        assert skips == trace['per_branch']['uncond']['reuse_path'] and len(skips) == int(k)
        p = ''.join('1' if i in skips else '0' for i in range(50))
        r.update(late25=p[25:].count('1'),longest_late25=max(map(len,p[25:].split('0'))),skip_path=p)
        old_index[sid,k] = r
    paired = []
    for a in agg:
        rr = [r for r in detail if r['group'] == a['group'] and r['k'] == a['k']]
        d = dict(group=a['group'],k=a['k'])
        for key in METRICS + ['late25','longest_late25']:
            diffs = [float(r[key])-float(old_index[r['sample_id'],r['k']][key]) for r in rr]
            d['delta_'+key] = st.fmean(diffs)
            if key in METRICS:
                d['wins_'+key] = sum(x < 0 if key == METRICS[2] else x > 0 for x in diffs)
        paired.append(d)
    oldagg = []
    for k in ['23','29','35']:
        rr = [r for r in old if r['k'] == k]
        oldagg.append(dict(k=k,n=10,**{key:st.fmean(float(r[key]) for r in rr) for key in METRICS+['late25','longest_late25','candidate_seconds']},
            latency_speedup=sum(float(r['baseline_seconds']) for r in rr)/sum(float(r['candidate_seconds']) for r in rr)))
    OUT.mkdir(exist_ok=True)
    (OUT / 'README.md').write_text('# Final paired readout\n\nExisting eight-run suite plus archived Dynamics128 e391 restricted to the identical ten prompts. No new inference.\n\n- e391_same10_per_video.csv / e391_same10_summary.csv: matched reference metrics and traces.\n- paired_vs_e391.csv: condition mean deltas and prompt-level wins; LPIPS lower is better.\n- VALIDATION.json: independently checked source hashes, aggregates and baseline video identity.\n\nOnly ten prompts and one seed; no significance or single-parameter causal claims. SEA7 comparisons also change the feature scheme.\n')
    writecsv(OUT/'e391_same10_per_video.csv',old); writecsv(OUT/'e391_same10_summary.csv',oldagg); writecsv(OUT/'paired_vs_e391.csv',paired)
    seal(Path(__file__))
    (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',new_candidates=240,e391_matched_candidates=30,conditions=24,
        suite_evidence_count=len(validation['evidence_sha256']),same_prompt_gpu_and_baseline_video_sha=True,
        quality_visual_qa='Actual quality_comparison.png inspected; labels, legends and six panels readable.',
        evidence_sha256=evidence),indent=2)+'\n')
    print(json.dumps(oldagg,indent=2))

if __name__ == '__main__': main()
