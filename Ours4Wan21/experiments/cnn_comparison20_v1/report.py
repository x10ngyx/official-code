import csv,json,hashlib,math,time
from pathlib import Path
from collections import defaultdict
EXP=Path('/mnt/hdd/xiongyuxiang/tmp/exp')
ROOT=EXP/'ours21_cnn_vbench20_v1'
OLD=EXP/'ours21_increase_vbench20_v1/analysis/all_methods20'
OUT=ROOT/'analysis/comparison20'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 out=list(csv.DictReader(p.open()))
 for r in out:
  for k,v in r.items():
   if k not in ('group','method','setting','sample_id'):
    try:r[k]=float(v)
    except (ValueError,TypeError):pass
 return out
def save(p,rr):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
while not (ROOT/'COMPLETE.json').exists():
 if (ROOT/'FAILED.json').exists():raise RuntimeError(read(ROOT/'FAILED.json'))
 time.sleep(15)
complete=read(ROOT/'COMPLETE.json')
assert complete['candidates']==240 and complete['paired_frames']==19440
for f,key in [('config.json','config_sha256'),('results.csv','results_sha256'),('per_video.csv','detail_sha256')]:assert sha(ROOT/f)==complete[key]
val=read(OLD/'VALIDATION.json');assert val['status']=='pass' and val['same20'] and val['same_baseline_sha']
for p,h in val['evidence_sha256'].items():assert sha(p)==h,p
cfg=read(ROOT/'config.json');ids={r['sample_id'] for r in cfg['prompts']}
raw=rows(ROOT/'per_video.csv');old=rows(OLD/'per_video.csv');metrics=['psnr_rgb_db','yuv_611_db','ssim_rgb','lpips_alex_v0_1_spatial']
fields=['baseline_seconds','generate_seconds','dit_cuda_seconds','t5_cuda_seconds','vae_decode_cuda_seconds','dit_tflops','estimated_t5_tflops_per_video','estimated_vae_decode_tflops_per_video',*metrics]
detail=[]
for r in raw:detail.append(dict(method=r['group'],setting=f"K{int(r['k'])}",sample_id=r['sample_id'],**{k:r[k] for k in fields}))
detail += [{k:r[k] for k in ['method','setting','sample_id',*fields]} for r in old if r['method'] in ('e391','SeaCache','Increase')]
assert len(detail)==400
buckets=defaultdict(list)
for r in detail:
 assert all(math.isfinite(r[k]) for k in fields)
 buckets[r['method'],r['setting']].append(r)
summary=[]
for (method,setting),rr in buckets.items():
 assert len(rr)==20 and {r['sample_id'] for r in rr}==ids
 for r in rr:
  base=next(x['baseline_seconds'] for x in raw if x['sample_id']==r['sample_id'])
  assert abs(r['baseline_seconds']-base)<1e-8
 summary.append(dict(method=method,setting=setting,n=20,speedup=sum(r['baseline_seconds'] for r in rr)/sum(r['generate_seconds'] for r in rr),**{k:sum(r[k] for r in rr)/20 for k in fields}))
for original in rows(ROOT/'results.csv'):
 s=next(s for s in summary if s['method']==original['group'] and s['setting']==f"K{int(original['k'])}")
 for k in ['speedup',*metrics,'generate_seconds']:assert abs(s[k]-original[k])<1e-8
for original in rows(OLD/'summary.csv'):
 if original['method'] not in ('e391','SeaCache','Increase'):continue
 s=next(s for s in summary if s['method']==original['method'] and s['setting']==original['setting'])
 for k in ['speedup',*fields]:assert abs(s[k]-original[k])<1e-8
paired=[];delta=[]
for g in cfg['groups']:
 for k,target in [(23,'1.8'),(29,'2.4'),(35,'3.0')]:
  a=buckets[g,f'K{k}']
  for method,setting in [('e391',f'K{k}'),('SeaCache','target'+target),('Increase','target'+target)]:
   if (method,setting) not in buckets:continue
   b={r['sample_id']:r for r in buckets[method,setting]}
   dd=[]
   for r in a:
    q=b[r['sample_id']];d=dict(group=g,k=k,reference_method=method,reference_setting=setting,sample_id=r['sample_id'],**{'delta_'+m:r[m]-q[m] for m in metrics});dd.append(d);paired.append(d)
   delta.append(dict(group=g,k=k,reference_method=method,reference_setting=setting,latency_change_pct=(sum(r['generate_seconds'] for r in a)/sum(r['generate_seconds'] for r in b.values())-1)*100,**{'delta_'+m:sum(d['delta_'+m] for d in dd)/20 for m in metrics},rgb_wins=sum(d['delta_psnr_rgb_db']>0 for d in dd)))
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'README.md').write_text('同20prompt的四组CNN、e391、SeaCache、Increase比较。summary.csv含完整组件时间/TFLOPs；per_video.csv与paired_deltas.csv保留逐prompt证据；无新增推理或速度标定。\n')
save(OUT/'summary.csv',summary);save(OUT/'per_video.csv',detail);save(OUT/'paired_deltas.csv',paired);save(OUT/'delta_summary.csv',delta)
lines=['# CNN与历史同VBench20的RGB PSNR对照','','240 CNN候选、19440对帧已完整完成。四组训练均200epoch，tau=.7/beta=1.5/cap30，选中G1 e182/G2 e175/G3 e192/G4 e173。三分支：直接池化[16,4,8,8]、全时间均值后池化[16,8,8]、全时间方差后池化[16,8,8]。G1原始三状态，G2原始两差值，G3 SEA三状态，G4 SEA两差值。','','同20prompt、同baseline、固定seed42。e391的训练数据/轮数和架构均与本次不同，因此不能将差异单独归因于CNN或feature。SeaCache低档无同20结果；中档与Increase高档存在实速差，按名义档位列出而非等速结论。未计算VBench score。']
for k,t in [(23,'1.8'),(29,'2.4'),(35,'3.0')]:
 lines+=['',f'## K{k} / 名义{t}档','','|方法|实际加速×|秒/视频|RGB PSNR↑|','|---|---:|---:|---:|']
 for method in ['G1','G2','G3','G4','e391','SeaCache','Increase']:
  setting=f'K{k}' if method.startswith('G') or method=='e391' else 'target'+t
  rr=[r for r in summary if r['method']==method and r['setting']==setting]
  if not rr:continue
  r=rr[0];lines.append(f"|{method}|{r['speedup']:.4f}|{r['generate_seconds']:.3f}|{r[metrics[0]]:.4f}|")
lines+=['','## CNN减历史对照','','|组|K|对照|耗时变化%|ΔRGB PSNR|RGB胜出/20|','|---|---:|---|---:|---:|---:|']
for r in delta:lines.append(f"|{r['group']}|{r['k']}|{r['reference_method']}|{r['latency_change_pct']:+.2f}|{r['delta_psnr_rgb_db']:+.4f}|{r['rgb_wins']}|")
lines+=['','组件时间/TFLOPs保留在summary.csv。总推理耗时含CNN与特征开销；DiT TFLOPs为实际block trace计数。单seed固定20prompt仅作描述性比较，非统计显著性或新测试集泛化结论。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
evidence={str(p):sha(p) for p in [ROOT/'COMPLETE.json',ROOT/'results.csv',ROOT/'per_video.csv',OLD/'VALIDATION.json',OLD/'summary.csv',OLD/'per_video.csv',Path(__file__)]}
(OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',conditions=len(summary),videos=len(detail),same20=True,same_baseline_seconds=True,old_evidence_hashes_verified=len(val['evidence_sha256']),evidence_sha256=evidence),indent=2)+'\n')
print((OUT/'REPORT.md').read_text(),flush=True)
