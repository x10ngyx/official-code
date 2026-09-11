import csv,json,hashlib,statistics as st,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'increase500_iql2_v1'))
from common import FIELDS,METRICS
E=Path('/mnt/hdd/xiongyuxiang/tmp/exp');R=E/'ours21_increase500_iql2_v1';OUT=R/'analysis/final_comparison20'
S={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def seal(p,expected=None):
 p=Path(p);h=sha(p)
 if expected is not None:assert h==expected,str(p)
 S[str(p)]=h
 return p
def read(p):return json.loads(seal(p).read_text())
def rows(p):return list(csv.DictReader(seal(p).open()))
def write(name,rr):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
 done=read(R/'COMPLETE.json');assert done['candidates']==120 and done['paired_frames']==9720
 for name,key in [('results.csv','results_sha256'),('per_video.csv','detail_sha256'),('config.json','config_sha256')]:seal(R/name,done[key])
 cfg=read(R/'config.json');prompts={p['sample_id']:p for p in cfg['prompts']};assert len(prompts)==20
 ref=Path(cfg['reference']);baseline={sid:read(ref/'baselines'/sid/'generation.json') for sid in prompts}
 detail=rows(R/'per_video.csv');assert len(detail)==120
 qs={}
 for g in range(4):
  summary=read(R/f'evaluation/quality/gpu{g}/summary.json')
  qr=rows(R/f'evaluation/quality/gpu{g}/per_video.csv');assert summary['video_count']==len(qr)
  for q in qr:
   for side in ['reference','candidate']:seal(q[side],q[side+'_sha256'])
   assert (int(q['frames']),int(q['width']),int(q['height']))==(81,832,480)
   qs[q['video_id']]=q
 assert len(qs)==120
 unified=[]
 for d in detail:
  q=qs[f"{d['group']}_K{d['k']}_{d['sample_id']}"]
  seal(ref/'baselines'/d['sample_id']/'video.mp4',q['reference_sha256'])
  for m in METRICS:assert abs(float(q[m+'_mean'])-float(d[m]))<1e-10
  gen=read(R/f"evaluation/candidates/{d['group']}/K{d['k']}/{d['sample_id']}/generation.json")
  assert gen['gpu_uuid']==baseline[d['sample_id']]['gpu_uuid'] and gen['protocol']==cfg['protocol']
  unified.append(dict(method=d['group'],setting='K'+d['k'],sample_id=d['sample_id'],baseline_seconds=float(d['baseline_seconds']),**{k:float(d[k]) for k in (*FIELDS,*METRICS)}))
 for a in rows(R/'results.csv'):
  rr=[d for d in unified if d['method']==a['group'] and d['setting']=='K'+a['k']]
  assert len(rr)==20 and {d['sample_id'] for d in rr}==set(prompts)
  for m in (*FIELDS,*METRICS):assert abs(st.fmean(d[m] for d in rr)-float(a[m]))<1e-8
  assert abs(sum(d['baseline_seconds'] for d in rr)/sum(d['generate_seconds'] for d in rr)-float(a['speedup']))<1e-10
 old=E/'ours21_dynamics128_e391_vbench50_random42_4gpu_v1'
 sources=[('e391',old,None),('SeaCache',E/'seacache_wan21_vbench50_speed24_4gpu_v1','2.4'),('SeaCache',E/'seacache_wan21_vbench50_speed30_4gpu_v1','3.0')]
 for method,root,target in sources:
  read(root/'COMPLETE.json')
  val=read(root/('analysis/QUALITY_VALIDATION.json' if method=='e391' else 'analysis/VALIDATION.json'))
  for p,h in val['evidence_sha256'].items():seal(root/p,h)
  for d in rows(root/'analysis/per_video.csv'):
   sid=d['sample_id']
   if sid not in prompts:continue
   setting='K'+d['k'] if method=='e391' else 'target'+target
   folder=root/f"shards/gpu{d['gpu']}"/('K'+d['k'] if method=='e391' else 'seacache')
   run=read(folder/'run.json');assert run['gpu_uuid']==baseline[sid]['gpu_uuid'] and run['protocol']==cfg['protocol']
   p=next(p for p in run['prompts'] if p['sample_id']==sid)
   assert p.get('prompt',p.get('prompt_en'))==prompts[sid].get('prompt',prompts[sid].get('prompt_en'))
   q=next(q for q in rows(folder/'quality/per_video.csv') if q['video_id']==sid)
   for side in ['reference','candidate']:seal(q[side],q[side+'_sha256'])
   seal(ref/'baselines'/sid/'video.mp4',q['reference_sha256'])
   assert (int(q['frames']),int(q['width']),int(q['height']))==(81,832,480)
   for m in METRICS:assert abs(float(d[m])-float(q[m+'_mean']))<1e-10
   c=next(c for c in read(folder/'components.json')['rows'] if c['sample_id']==sid)
   assert all(key in c for key in FIELDS[:7])
   assert abs(c['generate_seconds']-float(d['candidate_seconds']))<1e-8
   unified.append(dict(method=method,setting=setting,sample_id=sid,baseline_seconds=float(d['baseline_seconds']),**{k:float(c.get(k,0)) for k in FIELDS},**{k:float(d[k]) for k in METRICS}))
 assert len(unified)==220
 agg=[]
 for method,setting in dict.fromkeys((d['method'],d['setting']) for d in unified):
  rr=[d for d in unified if (d['method'],d['setting'])==(method,setting)]
  assert len(rr)==20 and {d['sample_id'] for d in rr}==set(prompts)
  agg.append(dict(method=method,setting=setting,n=20,speedup=sum(d['baseline_seconds'] for d in rr)/sum(d['generate_seconds'] for d in rr),**{k:st.fmean(d[k] for d in rr) for k in (*FIELDS,*METRICS)}))
 paired=[]
 for group in ['conservative','aggressive']:
  for k in [23,29,35]:
   for comparator,setting in [('e391','K'+str(k))]+([('SeaCache','target'+{29:'2.4',35:'3.0'}[k])] if k!=23 else []):
    a={d['sample_id']:d for d in unified if d['method']==group and d['setting']=='K'+str(k)}
    b={d['sample_id']:d for d in unified if d['method']==comparator and d['setting']==setting}
    row=dict(group=group,k=k,comparator=comparator,comparator_setting=setting,n=20)
    for m in METRICS:
     ds=[a[sid][m]-b[sid][m] for sid in prompts]
     row['delta_'+m]=st.fmean(ds);row['wins_'+m]=sum(x<0 if m==METRICS[-1] else x>0 for x in ds)
    row['latency_change_pct']=100*(sum(x['generate_seconds'] for x in a.values())/sum(x['generate_seconds'] for x in b.values())-1)
    paired.append(row)
 OUT.mkdir(parents=True,exist_ok=True)
 write('per_video.csv',unified);write('summary.csv',agg);write('paired_deltas.csv',paired)
 text=['# 同20个VBench prompt最终比较','', '保守e374；激进e358；两组mixed3500训练均400轮，验证集自动选点。120新候选/9720对帧质量完成。原生baseline、e391及SeaCache均复用；同prompt、同GPU UUID、同baseline视频SHA核验。','', '速度=同20条baseline总推理时间/候选总推理时间；质量为逐视频均值。PSNR/SSIM越高越好，LPIPS越低越好。','', '|方法|设置|实速×|推理秒|DiT TFLOPs|PSNR dB|SSIM|LPIPS|','|---|---|---:|---:|---:|---:|---:|---:|']
 for a in agg:text.append(f"|{a['method']}|{a['setting']}|{a['speedup']:.4f}|{a['generate_seconds']:.3f}|{a['dit_tflops']:.3f}|{a[METRICS[0]]:.4f}|{a[METRICS[1]]:.5f}|{a[METRICS[2]]:.5f}|")
 text+=['','## 配对差异：新组减对照','', '|组|K|对照|ΔPSNR|ΔSSIM|ΔLPIPS|PSNR胜出/20|耗时变化%|','|---|---|---|---:|---:|---:|---:|---:|']
 for a in paired:text.append(f"|{a['group']}|{a['k']}|{a['comparator']} {a['comparator_setting']}|{a['delta_'+METRICS[0]]:+.4f}|{a['delta_'+METRICS[1]]:+.5f}|{a['delta_'+METRICS[2]]:+.5f}|{a['wins_'+METRICS[0]]}|{a['latency_change_pct']:+.2f}|")
 text+=['','## 主要结论','', 'K23：激进组PSNR与e391基本相同，SSIM/LPIPS均改善；保守组PSNR降低0.1162dB，但SSIM/LPIPS略改善。','', 'K29：保守组三项质量均优于e391，PSNR提高0.1327dB；激进组三项均退步，PSNR降低0.3800dB。两组相较SeaCache中档质量较好，但耗时增加约10%。','', 'K35：激进组三项质量均优于e391与SeaCache，PSNR分别提高0.5841/0.4535dB，对两者均15/20条PSNR更高；保守组三项优于e391，但相较SeaCache的SSIM略低。','', '这组结果支持按预算选模型：K23激进、K29保守、K35激进；这属于测试后观察，未据此重选checkpoint。没有跨三档统一最优的新训练组。','', 'SeaCache中档约2.45×与模型K29约2.23×不等速；K35约3.00×与SeaCache约3.01×较接近。未发现同20prompt的SeaCache1.8档存档，不补造该档结论。','', '单seed描述性结果，不作统计显著性结论；相较原e391同时改变了数据和IQL训练设置，不能把差异单独归因于increase数据。VBench20指prompt子集，不是VBench score。T5/DiT/VAE等组件明细保留在CSV。']
 (OUT/'REPORT.md').write_text('\n'.join(text)+'\n');(OUT/'README.md').write_text((HERE/'README.md').read_text())
 seal(__file__)
 (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',new_candidates=120,archived_e391=60,archived_seacache=40,conditions=11,same20=True,same_gpu_and_baseline_sha=True,evidence_sha256=S),indent=2)+'\n')
 print('\n'.join(text))
if __name__=='__main__':main()
