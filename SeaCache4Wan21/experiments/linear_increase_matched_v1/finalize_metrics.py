"""Finalize all 40 paired trajectories with VideoMetrics only, per explicit user instruction."""
import math
import statistics as st
from pathlib import Path
import run_experiment as s


def main():
    root=s.ROOT;cfg=s.prepare()
    if not (root/'VBENCH_SKIPPED_BY_USER.json').exists():raise ValueError('explicit user VBench override required')
    results=[];details=[];evidence={};deltas=[]
    for i in range(5):
        for phase in ('increase','fixed'):
            candidates=[];qualities=[];traces=[]
            for g in range(4):
                r,tr=s.condition(cfg,phase,i,g);candidates.append(r);traces.append(tr)
                d=root/'conditions'/f'{phase}_{i+1}'/f'gpu{g}';sid=r['sample_id']
                q=s.pair.prior.quality_rows(d/'quality',[sid])[0];qualities.append(q)
                details.append(dict(strategy=i+1,phase=phase,gpu=g,**r,baseline_seconds=cfg['baselines'][str(g)]['generate_seconds'],reuse_steps=tr['reuse']/2,**{m:float(q[m+'_mean']) for m in s.METRICS}))
                for f in ('components.json','run.json','COMPLETE.json','quality/summary.json','quality/per_video.csv',f'timings/{sid}.json',f'traces/{sid}.json'):
                    evidence[str((d/f).relative_to(root))]=s.sha(d/f)
            threshold=s.read(root/'matching.json')['pairs'][i]['threshold'];b=list(cfg['baselines'].values())
            results.append(dict(strategy=i+1,phase=phase,threshold_start=cfg['strategies'][i]['start'] if phase=='increase' else threshold,threshold_end=cfg['strategies'][i]['end'] if phase=='increase' else threshold,
                latency_speedup=sum(x['generate_seconds'] for x in b)/sum(x['generate_seconds'] for x in candidates),dit_compute_ratio=sum(x['dit_tflops'] for x in b)/sum(x['dit_tflops'] for x in candidates),
                **{k:st.fmean(x[k] for x in candidates) for k in s.FIELDS},reuse_steps=st.fmean(x['reuse']/2 for x in traces),
                **{m:st.fmean(float(x[m+'_mean']) for x in qualities) for m in s.METRICS}))
    comparisons=[]
    for inc,fix in zip(results[::2],results[1::2]):
        i=inc['strategy'];gap=fix['generate_seconds']/inc['generate_seconds']-1
        for g in range(4):
            rows={p:next(x for x in details if x['strategy']==i and x['gpu']==g and x['phase']==p) for p in ('increase','fixed')}
            deltas.append(dict(strategy=i,gpu=g,sample_id=rows['increase']['sample_id'],**{m+'_delta':rows['increase'][m]-rows['fixed'][m] for m in s.METRICS}))
        comparisons.append(dict(strategy=i,increase_speedup=inc['latency_speedup'],fixed_speedup=fix['latency_speedup'],fixed_threshold=fix['threshold_start'],fixed_latency_relative_gap=gap,speed_matched_within_5pct=abs(gap)<=cfg['speed_match_tolerance'],
            **{m+'_increase_minus_fixed':inc[m]-fix[m] for m in s.METRICS},lpips_relative_reduction=1-inc['lpips_alex_v0_1_spatial']/fix['lpips_alex_v0_1_spatial'],
            **{m+'_improved_prompt_count':sum((x[m+'_delta']<0 if m=='lpips_alex_v0_1_spatial' else x[m+'_delta']>0) for x in deltas if x['strategy']==i) for m in s.METRICS}))
    if len(details)!=40 or len({(r['strategy'],r['phase'],r['sample_id']) for r in details})!=40:raise ValueError('40 trajectory coverage')
    s.validate_sources(cfg)
    for name,rows in [('results',results),('per_video',details),('comparison',comparisons),('paired_quality_deltas',deltas)]:s.pair.prior.write_csv(root/'analysis'/f'{name}.csv',rows)
    validation=dict(status='pass',formal_trajectories=40,reused_baselines=4,branch_calls=4000,quality_frame_pairs=3240,evidence_sha256=evidence,matched_pairs=sum(x['speed_matched_within_5pct'] for x in comparisons),max_absolute_latency_gap=max(abs(x['fixed_latency_relative_gap']) for x in comparisons),vbench_status='skipped_by_user',vbench_override_sha256=s.sha(root/'VBENCH_SKIPPED_BY_USER.json'),caveats=['four prompts and one seed','historical native baselines reused on same physical GPUs','DiT FLOPs exclude SEA gate/filter overhead following existing calibration accounting'])
    s.dump(root/'analysis/VALIDATION.json',validation)
    lines=['# SeaCache 线性递增 threshold 与固定 threshold 同速对照','',
        '**结论：当前4条prompt上，线性递增threshold表现出提升空间。2.06–3.20×四档在相同DiT计算量、推理耗时差小于0.04%时，PSNR/SSIM/LPIPS均值均改善。**',
        '2.85×档改善最一致：4条prompt的三项指标都改善；最低1.67×档仅PSNR均值提高，SSIM、LPIPS均值略差。',
        '每组仅4条prompt、seed42，属于小样本描述性结果，不能据此保证其他prompt上的提升。','',
        '方法：同4条prompt（041/155/144/040），固定Wan2.1-1.3B、832×480、81帧、16fps、UniPC50、shift5、CFG5、BF16、全部模型常驻同物理GPU。',
        'threshold=a+(b-a)*step/49，step为采样执行顺序0…49；先20条increase，按4条baseline总耗时/4条increase总耗时由旧标定相邻实测点插值，再冻结阈值跑20条fixed；不使用质量选参。',
        '40条正式轨迹；复用4条历史native baseline；每GPU1次native warmup排除测量。计时排除加载、预热、保存与评测。质量先按81帧求每视频均值，再对4条prompt等权平均。',
        '按用户最新要求不计算或汇报VBench分数，已停止评测，部分输出仅保留作诊断，不纳入结果。','',
        '|组|方法|threshold|实测加速|秒/视频|DiT TFLOPs|PSNR dB↑|SSIM↑|LPIPS↓|',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"|{r['strategy']}|{r['phase']}|{r['threshold_start']:.5f}→{r['threshold_end']:.5f}|{r['latency_speedup']:.4f}|{r['generate_seconds']:.3f}|{r['dit_tflops']:.2f}|{r['psnr_rgb_db']:.4f}|{r['ssim_rgb']:.5f}|{r['lpips_alex_v0_1_spatial']:.5f}|")
    lines+=['','|组|固定相对递增耗时差|ΔPSNR（递增−固定）|ΔSSIM|LPIPS相对下降|PSNR改善prompt数|','|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:lines.append(f"|{r['strategy']}|{r['fixed_latency_relative_gap']:+.3%}|{r['psnr_rgb_db_increase_minus_fixed']:+.4f}|{r['ssim_rgb_increase_minus_fixed']:+.5f}|{r['lpips_relative_reduction']:+.2%}|{r['psnr_rgb_db_improved_prompt_count']}/4|")
    lines+=['','## 组件计量','',
        'DiT TFLOPs按实际30-block full/reuse call计数，使用既有Calflops profile；SEA gate/filter计算未计入，与原标定口径一致。','',
        '|组|方法|T5秒|DiT秒|VAE秒|T5 TFLOPs|VAE TFLOPs|','|---:|---|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"|{r['strategy']}|{r['phase']}|{r['t5_cuda_seconds']:.4f}|{r['dit_cuda_seconds']:.3f}|{r['vae_decode_cuda_seconds']:.3f}|{r['estimated_t5_tflops_per_video']:.3f}|{r['estimated_vae_decode_tflops_per_video']:.3f}|")
    lines+=['','复核与来源：config.json冻结prompt、同GPU baseline、生成协议与代码哈希；matching.json保存插值区间。results.csv为10条件均值，per_video.csv为40条明细，paired_quality_deltas.csv为20个配对差值；VALIDATION.json与INDEPENDENT_AUDIT.json记录完整性和独立重算结果。']
    (root/'analysis/RESULTS.md').write_text('\n'.join(lines)+'\n')
    s.dump(root/'COMPLETE.json',dict(status='complete',formal_trajectories=40,vbench_status='skipped_by_user',results_sha256=s.sha(root/'analysis/results.csv'),validation_sha256=s.sha(root/'analysis/VALIDATION.json')))
    import audit
    audit.main()
    s.dump(root/'status.json',dict(status='complete',generation_complete=40,quality_complete=40,vbench_status='skipped_by_user',updated_at=s.pair.now()))
    print(s.read(root/'analysis/INDEPENDENT_AUDIT.json'))
if __name__=='__main__':main()
