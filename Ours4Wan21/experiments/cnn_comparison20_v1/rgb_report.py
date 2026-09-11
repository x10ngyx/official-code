import csv,json,hashlib,os,sys,subprocess,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'experiments/cnn_vbench20_v1'))
from common import audit_trace
EXP=Path('/mnt/hdd/xiongyuxiang/tmp/exp');ROOT=EXP/'ours21_cnn_vbench20_v1';OLD=EXP/'ours21_increase_vbench20_v1/analysis/all_methods20';OUT=ROOT/'analysis/rgb_comparison20'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open()))
def save(p,rr):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
OUT.mkdir(parents=True,exist_ok=True);(OUT/'README.md').write_text('RGB PSNR主对比，官方VideoMetrics仅选择psnr；完整240视频/19440帧，历史同20prompt，逐视频组件与配对差值见CSV。\n')
def quality(i):
 out=OUT/f'quality_gpu{i}';d=ROOT/f'evaluation/quality_inputs/gpu{i}'
 if (out/'summary.json').exists():return
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 with (OUT/f'quality_gpu{i}.log').open('w') as f:
  subprocess.run([sys.executable,str(PROJECT.parent/'VideoMetrics/evaluate.py'),'--reference-dir',str(d/'reference'),'--candidate-dir',str(d/'candidate'),'--metrics','psnr','--device','cpu','--expected-frames','81','--output-dir',str(out)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
with ThreadPoolExecutor(4) as p:list(p.map(quality,range(4)))
cfg=read(ROOT/'config.json');picks=read(ROOT/'SELECTED.json');ref=Path(cfg['reference']);ids={r['sample_id'] for r in cfg['prompts']};v=read(OLD/'VALIDATION.json')
assert v['status']=='pass'
for p,h in v['evidence_sha256'].items():assert sha(p)==h,p
for sid in ids:
 p=ref/'baselines'/sid/'video.mp4';assert sha(p)==v['evidence_sha256'][str(p)]
fields=['baseline_seconds','generate_seconds','dit_cuda_seconds','t5_cuda_seconds','vae_decode_cuda_seconds','dit_tflops','estimated_t5_tflops_per_video','estimated_vae_decode_tflops_per_video','psnr_rgb_db']
detail=[];sealed=0;frames=0
for i in range(4):
 qq=rows(OUT/f'quality_gpu{i}/per_video.csv');summary=read(OUT/f'quality_gpu{i}/summary.json');frames+=summary['frame_count_total'];assert len(qq)==summary['video_count']
 qs={q['video_id']:q for q in qq}
 for j in read(ROOT/f'jobs/gpu{i}.json'):
  p=Path(j['output']);marker=read(p/'COMPLETE.json');assert marker['identity']['job']==j
  for f,h in marker['files'].items():assert sha(p/f)==h;sealed+=1
  assert j['checkpoint']==dict(path=picks[j['group']]['checkpoint'],sha256=picks[j['group']]['sha256'])
  g=read(p/'generation.json');assert g['protocol']==cfg['protocol'] and g['gpu_uuid']==j['expected_gpu_uuid']
  audit_trace(read(p/'trace.json'),read(p/'timing.json'),j['skip_budget'])
  q=qs[f"{j['group']}_K{j['skip_budget']}_{j['sample_id']}"];assert (int(q['frames']),int(q['width']),int(q['height']))==(81,832,480)
  assert Path(q['reference']).resolve()==(ref/'baselines'/j['sample_id']/'video.mp4').resolve()
  assert Path(q['candidate']).resolve()==(p/'video.mp4').resolve()
  for side in ['reference','candidate']:assert sha(q[side])==q[side+'_sha256']
  m=read(p/'measurement.json');base=read(ref/'baselines'/j['sample_id']/'measurement.json')['generate_seconds']
  detail.append(dict(method=j['group'],setting=f"K{j['skip_budget']}",sample_id=j['sample_id'],baseline_seconds=base,**{k:m[k] for k in fields if k not in ['baseline_seconds','psnr_rgb_db']},psnr_rgb_db=float(q['psnr_rgb_db_mean'])))
assert len(detail)==240 and frames==19440
for r in rows(OLD/'per_video.csv'):
 if r['method'] in ['e391','SeaCache','Increase','conservative','aggressive']:detail.append({**{k:r[k] for k in ['method','setting','sample_id']},**{k:float(r[k]) for k in fields}})
buckets=defaultdict(list)
for r in detail:
 assert all(math.isfinite(r[k]) for k in fields);buckets[r['method'],r['setting']].append(r)
summary=[]
for (method,setting),rr in buckets.items():
 assert len(rr)==20 and {r['sample_id'] for r in rr}==ids
 for r in rr:assert abs(r['baseline_seconds']-read(ref/'baselines'/r['sample_id']/'measurement.json')['generate_seconds'])<1e-8
 summary.append(dict(method=method,setting=setting,n=20,speedup=sum(r['baseline_seconds'] for r in rr)/sum(r['generate_seconds'] for r in rr),**{k:sum(r[k] for r in rr)/20 for k in fields}))
for r in rows(OLD/'summary.csv'):
 if r['method'] not in ['e391','SeaCache','Increase','conservative','aggressive']:continue
 s=next(s for s in summary if (s['method'],s['setting'])==(r['method'],r['setting']))
 for k in ['speedup',*fields]:assert abs(s[k]-float(r[k]))<1e-8
paired=[];deltas=[]
for group in cfg['groups']:
 for k,t in [(23,'1.8'),(29,'2.4'),(35,'3.0')]:
  a=buckets[group,f'K{k}']
  for method,setting in [('e391',f'K{k}'),('conservative',f'K{k}'),('aggressive',f'K{k}'),('SeaCache','target'+t),('Increase','target'+t)]:
   if (method,setting) not in buckets:continue
   b={r['sample_id']:r for r in buckets[method,setting]};dd=[]
   for r in a:
    d=dict(group=group,k=k,reference=method,sample_id=r['sample_id'],rgb_psnr_delta=r['psnr_rgb_db']-b[r['sample_id']]['psnr_rgb_db']);paired.append(d);dd.append(d)
   deltas.append(dict(group=group,k=k,reference=method,rgb_psnr_delta=sum(d['rgb_psnr_delta'] for d in dd)/20,rgb_wins=sum(d['rgb_psnr_delta']>0 for d in dd),latency_change_pct=(sum(r['generate_seconds'] for r in a)/sum(r['generate_seconds'] for r in b.values())-1)*100))
save(OUT/'summary.csv',summary);save(OUT/'per_video.csv',detail);save(OUT/'paired_deltas.csv',paired);save(OUT/'delta_summary.csv',deltas)
lines=['# CNN与e391、SeaCache、Increase：RGB PSNR对比','','240 CNN候选/19440对帧全部完成RGB PSNR。官方VideoMetrics实现，先逐帧计算再逐视频等权平均；同20prompt/seed42/同GPU baseline。G1原始三状态(e182)，G2原始两差值(e175)，G3 SEA三状态(e192)，G4 SEA两差值(e173)。每组均含直接3D池化、全时间均值后2D池化、全时间方差后2D池化三个分支；mixed3500、200epoch、tau=.7/beta=1.5/cap30。']
for k,t in [(23,'1.8'),(29,'2.4'),(35,'3.0')]:
 lines+=['',f'## K{k} / 名义{t}档','','|方法|实际加速×|秒/视频|RGB PSNR(dB)↑|','|---|---:|---:|---:|']
 for method in ['G1','G2','G3','G4','e391','conservative','aggressive','SeaCache','Increase']:
  setting=f'K{k}' if method.startswith('G') or method in ['e391','conservative','aggressive'] else 'target'+t
  rr=[r for r in summary if r['method']==method and r['setting']==setting]
  if rr:
   r=rr[0];lines.append(f"|{method}|{r['speedup']:.4f}|{r['generate_seconds']:.3f}|{r['psnr_rgb_db']:.4f}|")
lines+=['','## CNN减对照','','|组|K|对照|ΔRGB PSNR(dB)|RGB胜出/20|耗时变化%|','|---|---:|---|---:|---:|---:|']
for d in deltas:lines.append(f"|{d['group']}|{d['k']}|{d['reference']}|{d['rgb_psnr_delta']:+.4f}|{d['rgb_wins']}|{d['latency_change_pct']:+.2f}|")
lines+=['','SeaCache低档无同20prompt结果。中档CNN/e391约2.23×、Increase约2.33×、SeaCache约2.45×；高档Increase仅2.70×而其他约3×，不视为等速比较。e391训练数据/轮数与本次不同，不能将收益单独归因于架构或feature。单seed固定20prompt，仅描述性比较。完整组件时间和TFLOPs保留在summary.csv；此次主结论仅依据RGB PSNR。']
lines[2:2]=['已补入同mixed3500的Dynamics128+MLP：conservative=e374，tau=.6/beta=1/cap20；aggressive=e358，tau=.9/beta=3/cap100，均训练400轮。本次CNN从同一transition来源读取相同2800/350/350划分，但仅200轮、tau=.7/beta=1.5/cap30，仍非架构或feature的单因素消融。G1/G2在三档均高于这两个同数据集MLP对照。','RGB PSNR结论：G2低/中档最佳，G1高档最佳。G1相对e391三档+0.6148/+0.2571/+1.1116dB；G2为+0.8729/+0.3785/+0.9150dB，实速接近。高档G1相对SeaCache+0.9810dB、17/20prompt胜出，耗时仅+0.17%。Increase低档仍高于G2 0.5943dB；中档低于G2 0.7028dB但耗时更短；高档高于G1 1.2076dB但耗时多约11.4%，不能视作等速胜出。SEA没有显示稳定收益：G3比G1高档下降约0.976dB，G4比G2三档均下降。固定20prompt单seed，不能直接推断泛化或特征单因素因果。','']
lines+=['','CNN三档DiT TFLOPs/视频为15390.606/11970.961/8551.316；T5估算9.690、VAE decode估算274.177 TFLOPs/视频。各组件实测时间均在summary.csv中，推理耗时包含特征提取及CNN决策。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
(OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',cnn_videos=240,paired_frames=frames,conditions=len(summary),same20=True,same_baseline_sha=True,sealed_files_verified=sealed,old_evidence_hashes_verified=len(v['evidence_sha256']),evidence_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'config.json',ROOT/'SELECTED.json',OLD/'VALIDATION.json',OLD/'summary.csv',OLD/'per_video.csv',OUT/'summary.csv',OUT/'per_video.csv']}),indent=2)+'\n')
crosschecked=0
for q in (ROOT/'evaluation/quality').glob('*/per_video.csv'):
 i=q.parent.name[-1];a={x['video_id']:float(x['psnr_rgb_db_mean']) for x in rows(q)}
 for x in rows(OUT/f'quality_gpu{i}/per_video.csv'):
  assert abs(a[x['video_id']]-float(x['psnr_rgb_db_mean']))<1e-8
  crosschecked+=1
validation=read(OUT/'VALIDATION.json');validation['full_metrics_rgb_crosscheck_videos']=crosschecked
(OUT/'VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
print((OUT/'REPORT.md').read_text(),flush=True)
