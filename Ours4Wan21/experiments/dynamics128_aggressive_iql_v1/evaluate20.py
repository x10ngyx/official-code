"""Evaluate the stable aggressive checkpoint on the unchanged online evaluation20."""
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from ours4wan21.contracts import EXP_ROOT,MODEL_ROOT,OFFICIAL,PROTOCOL,WORKSPACE,dump,sha256,create_nested_result
from ours4wan21.online_reference import load_evaluation_bundle,gpu_uuids
from run import ROOT,ANALYSIS,MODE

REFERENCE=EXP_ROOT/'ours21_online_eval20_from_vbench50_random42_v1'
OLD=EXP_ROOT/'ours21_dynamics128_e391_vbench50_random42_4gpu_v1'
FIELDS=('generate_seconds','dit_cuda_seconds','t5_cuda_seconds','vae_decode_cuda_seconds',
        'dit_tflops','estimated_t5_tflops_per_video','estimated_vae_decode_tflops_per_video',
        'predictor_tflops','predictor_network_cuda_seconds','predictor_decision_wall_seconds','latent_feature_wall_seconds')
METRICS=('psnr_rgb_db','ssim_rgb','lpips_alex_v0_1_spatial')


def js(p):return json.loads(Path(p).read_text())


def writecsv(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def freeze_plan():
    reference=load_evaluation_bundle(REFERENCE)
    assert reference['targets']==[1.8,2.4,3.0] and reference['skip_budgets']==[23,29,35]
    ids=[p['sample_id'] for p in reference['rows']]
    original=js(OLD/'config.json');uuids=gpu_uuids(['0','1','2','3'])
    shards={g:[dict(sample_id=r['sample_id'],prompt_en=r['prompt']) for r in reference['rows'] if r['baseline_gpu_uuid']==uuid]
            for g,uuid in enumerate(uuids)}
    assert sum(map(len,shards.values()))==20 and all(shards.values())
    for g,rows in shards.items():
        assert [r['sample_id'] for r in rows]==[sid for sid in ids if sid in original['shard_ids'][str(g)]]
    return reference,shards,uuids


def audit_trace(trace,timing,k):
    assert trace['total_steps']==50 and len(trace['decisions'])==len(timing['calls'])==100
    paths=[]
    for offset,branch in enumerate(('cond','uncond')):
        ds=[d for d in trace['decisions'] if d['branch']==branch]
        assert [d['step_index'] for d in ds]==list(range(50))
        a=[int(d['action']=='reuse') for d in ds]
        assert sum(a)==k
        for i,v in enumerate(a):
            assert timing['calls'][2*i+offset]['blocks_executed']==(0 if v else 30)
        paths.append(a)
    assert paths[0]==paths[1] and trace['step_reuse']==k
    late=paths[0][25:];best=run=0
    for v in late:run=run+1 if v else 0;best=max(best,run)
    return sum(late),best,''.join(map(str,paths[0]))


def main():
    ref,shards,uuids=freeze_plan()
    selected=js(ANALYSIS/'checkpoint_selection.json')
    assert 300<selected['checkpoint_epoch']<400 and sha256(selected['checkpoint'])==selected['checkpoint_sha256']
    out=ROOT/'evaluation20'
    config=dict(protocol=PROTOCOL,targets=ref['targets'],skip_budgets=ref['skip_budgets'],vbench_enabled=False,
        reference_bundle=str(REFERENCE),reference_manifest_sha256=sha256(REFERENCE/'manifest.json'),
        selected_checkpoint=selected['checkpoint'],checkpoint_sha256=selected['checkpoint_sha256'],
        selected_epoch=selected['checkpoint_epoch'],gpu_uuids=uuids,shards=shards,
        flops_profile=js(OLD/'config.json')['flops_profile'],prompt_ids=[r['sample_id'] for r in ref['rows']])
    create_nested_result(out,ROOT,'# Frozen online evaluation20\n\nSame 20 prompts and native baseline bundle as online fine-tuning. K23/K29/K35, three targets, no VBench score. shards/ holds measured candidate generation; prompts/ and references/ freeze same-GPU identity. results.csv/per_video.csv and trace_metrics.csv compare with existing e391 data for the same prompts.')
    dump(out/'config.json',config)
    for folder in ('prompts','references','shards','logs'):
        (out/folder).mkdir();(out/folder/'README.md').write_text(f'# {folder}\n\nFrozen online20 evaluation {folder}; see ../README.md.\n')
    for g,rows in shards.items():
        (out/'prompts'/f'gpu{g}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        d=out/'references'/f'gpu{g}';d.mkdir()
        (d/'README.md').write_text('# Same-GPU reference videos\n\nSymlinks into the sealed online evaluation20 baseline bundle.\n')
        for r in rows:(d/(r['sample_id']+'.mp4')).symlink_to(REFERENCE/'baselines'/r['sample_id']/'video.mp4')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',TORCH_HOME=str(MODEL_ROOT/'torch-cache'))
    def execute(label,cmd,g):
        with (out/'logs'/f'{label}.log').open('x') as f:
            subprocess.run(cmd,cwd=PROJECT,env=dict(env,CUDA_VISIBLE_DEVICES=uuids[g]),stdout=f,stderr=subprocess.STDOUT,check=True)
    def generate(g):
        parent=out/'shards'/f'gpu{g}';parent.mkdir();(parent/'README.md').write_text('# GPU shard\n\nK23/K29/K35 candidate videos, timing, components and trace.\n')
        for k in (23,29,35):
            d=parent/f'K{k}'
            execute(f'generate_gpu{g}_K{k}',[sys.executable,'generate.py',
                '--wan21-root',str(WORKSPACE/'data/source/Wan2.1-65386b2'),
                '--checkpoint-dir',str(MODEL_ROOT/'Wan2.1-T2V-1.3B'),
                '--prompts',str(out/'prompts'/f'gpu{g}.jsonl'),'--flops-profile',config['flops_profile'],
                '--output-dir',str(d),'--result-parent',str(ROOT),
                '--policy-checkpoint',selected['checkpoint'],'--state-mode',MODE,'--skip-budget',str(k)],g)
            run=js(d/'run.json');done=js(d/'COMPLETE.json')
            assert run['protocol']==PROTOCOL and run['gpu_uuid']==uuids[g]
            assert run['policy_sha256']==selected['checkpoint_sha256'] and run['skip_budget']==k
            assert done['status']=='generation_complete' and done['videos']==len(shards[g])
    def quality(g):
        for k in (23,29,35):
            d=out/'shards'/f'gpu{g}'/f'K{k}'
            execute(f'quality_gpu{g}_K{k}',[sys.executable,str(OFFICIAL/'VideoMetrics/evaluate.py'),
                '--reference-dir',str(out/'references'/f'gpu{g}'),'--candidate-dir',str(d/'videos'),
                '--expected-frames','81','--device','cuda:0','--model-cache',str(MODEL_ROOT/'torch-cache'),
                '--output-dir',str(d/'quality')],g)
    dump(out/'STATUS.json',dict(status='waiting_for_idle_gpus'))
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        time.sleep(30)
    dump(out/'STATUS.json',dict(status='generating'))
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(generate,range(4)))
    dump(out/'STATUS.json',dict(status='quality'))
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(quality,range(4)))
    old_quality={(int(r['k']),r['sample_id']):r for r in csv.DictReader((OLD/'analysis/per_video.csv').open())}
    detail=[];traces=[];evidence={}
    for g,rs in shards.items():
        for k in (23,29,35):
            d=out/'shards'/f'gpu{g}'/f'K{k}'
            component={r['sample_id']:r for r in js(d/'components.json')['rows']}
            qrows={r['video_id']:r for r in csv.DictReader((d/'quality/per_video.csv').open())}
            assert set(component)==set(qrows)=={r['sample_id'] for r in rs}
            for r in rs:
                sid=r['sample_id'];q=qrows[sid];c=component[sid]
                b=js(REFERENCE/'baselines'/sid/'measurement.json');old=old_quality[k,sid]
                assert (int(q['frames']),int(q['width']),int(q['height']))==(81,832,480)
                for side in ('reference','candidate'):assert sha256(q[side])==q[side+'_sha256']
                assert math.isclose(b['generate_seconds'],float(old['baseline_seconds']),abs_tol=1e-8)
                for key in FIELDS:assert math.isfinite(c[key]) and c[key]>=0
                for key in METRICS:assert math.isfinite(float(q[key+'_mean']))
                row=dict(k=k,target=ref['targets'][ref['skip_budgets'].index(k)],sample_id=sid,gpu=g,
                    baseline_seconds=b['generate_seconds'],**{key:c[key] for key in FIELDS},
                    **{key:float(q[key+'_mean']) for key in METRICS},
                    **{'e391_'+key:float(old[key]) for key in METRICS},e391_seconds=float(old['candidate_seconds']))
                detail.append(row)
                for label,folder in [('aggressive',d),('e391',OLD/'shards'/f'gpu{g}'/f'K{k}')]:
                    tp=folder/'traces'/f'{sid}.json';timep=folder/'timings'/f'{sid}.json'
                    late,run,path=audit_trace(js(tp),js(timep),k)
                    traces.append(dict(method=label,k=k,sample_id=sid,late25=late,longest_late25=run,skip_path=path))
                    evidence[str(tp)]=sha256(tp);evidence[str(timep)]=sha256(timep)
            for p in [d/'components.json',d/'run.json',d/'COMPLETE.json',d/'quality/per_video.csv']:
                evidence[str(p)]=sha256(p)
    assert len(detail)==60 and len(traces)==120
    results=[]
    for k,target in zip(ref['skip_budgets'],ref['targets']):
        rr=[r for r in detail if r['k']==k];assert len(rr)==20
        result=dict(k=k,target=target,n=20,
            latency_speedup=sum(r['baseline_seconds'] for r in rr)/sum(r['generate_seconds'] for r in rr),
            e391_latency_speedup=sum(r['baseline_seconds'] for r in rr)/sum(r['e391_seconds'] for r in rr),
            **{key:st.fmean(r[key] for r in rr) for key in (*FIELDS,*METRICS,*('e391_'+m for m in METRICS),'e391_seconds')})
        result['delta_psnr']=result['psnr_rgb_db']-result['e391_psnr_rgb_db']
        for method in ('aggressive','e391'):
            tt=[r for r in traces if r['method']==method and r['k']==k]
            result[method+'_late25_mean']=st.fmean(r['late25'] for r in tt)
            result[method+'_longest_late25_mean']=st.fmean(r['longest_late25'] for r in tt)
        results.append(result)
    writecsv(out/'per_video.csv',detail);writecsv(out/'results.csv',results);writecsv(out/'trace_metrics.csv',traces)
    lines=['# Aggressive Dynamics128：在线同20prompt三档评测','',
        f"选择epoch {selected['checkpoint_epoch']}；固定K23/K29/K35、同物理GPU旧baseline；不计算VBench分数。",'',
        '| 名义档 | 实测加速 | 推理秒 | PSNR | 原e391 PSNR | ΔPSNR | SSIM | LPIPS | DiT TFLOPs |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['target']:.1f}× | {r['latency_speedup']:.4f}× | {r['generate_seconds']:.3f} | {r['psnr_rgb_db']:.4f} | {r['e391_psnr_rgb_db']:.4f} | {r['delta_psnr']:+.4f} | {r['ssim_rgb']:.5f} | {r['lpips_alex_v0_1_spatial']:.5f} | {r['dit_tflops']:.3f} |")
    lines+=['','results.csv保留T5/VAE/DiT、predictor及feature的时间和TFLOPs；trace_metrics.csv包含同20条原e391与新策略的后段skip分配。',
        '原e391质量/计量复用原随机50实验中相同20条视频，baseline与prompt配对已核对；旧耗时非同时重测，质量仅单seed描述性比较。']
    lines+=['','| 名义档 | 激进组后25步skip | e391后25步skip | 激进组后段最长连续skip | e391后段最长连续skip |',
        '|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['target']:.1f}× | {r['aggressive_late25_mean']:.2f} | {r['e391_late25_mean']:.2f} | {r['aggressive_longest_late25_mean']:.2f} | {r['e391_longest_late25_mean']:.2f} |")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    assert sha256(REFERENCE/'manifest.json')==config['reference_manifest_sha256']
    dump(out/'VALIDATION.json',dict(status='pass',prompts=20,candidates=60,quality_frames=4860,
        raw_cfg_actions_validated=12000,baseline_reused=True,checkpoint_epoch=selected['checkpoint_epoch'],
        evidence_sha256=evidence,vbench_status='skipped_by_user'))
    done=dict(status='complete',prompts=20,candidates=60,results_sha256=sha256(out/'results.csv'),
        validation_sha256=sha256(out/'VALIDATION.json'),vbench_status='skipped_by_user',results=results)
    dump(out/'COMPLETE.json',done);dump(out/'STATUS.json',dict(status='complete'))


if __name__=='__main__':main()
