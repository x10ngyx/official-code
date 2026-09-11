import csv,json,hashlib,statistics as st,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
E=Path('/mnt/hdd/xiongyuxiang/tmp/exp');I=E/'ours21_increase_vbench20_v1';M=E/'ours21_increase500_iql2_v1';OLD=M/'analysis/final_comparison20';Y=M/'analysis/yuv_psnr20';OUT=I/'analysis/all_methods20'
S={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def seal(p,h=None):
 p=Path(p);value=sha(p)
 if h is not None:assert value==h,str(p)
 S[str(p)]=value;return p
def read(p):return json.loads(seal(p).read_text())
def rows(p):return list(csv.DictReader(seal(p).open()))
def write(name,rr):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
 done=read(I/'COMPLETE.json');seal(I/'analysis/results.csv',done['results_sha256']);seal(I/'VALIDATION.json',done['validation_sha256'])
 for f,h in read(I/'VALIDATION.json')['evidence_sha256'].items():seal(I/f,h)
 oldv=read(OLD/'VALIDATION.json')
 for f,h in oldv['output_sha256'].items():seal(OLD/f,h)
 for f,h in oldv['evidence_sha256'].items():seal(f,h)
 yv=read(Y/'VALIDATION.json')
 for f,h in yv['output_sha256'].items():seal(Y/f,h)
 cfg=read(I/'config.json');ids={p['sample_id'] for p in cfg['prompts']};assert len(ids)==20
 for f,h in cfg['source_sha256'].items():seal(f,h)
 raw=rows(OLD/'per_video.csv');yr={(d['method'],d['setting'],d['sample_id']):d for d in rows(Y/'per_video.csv')}
 fields=['baseline_seconds','generate_seconds','dit_cuda_seconds','t5_cuda_seconds','vae_decode_cuda_seconds','dit_tflops','estimated_t5_tflops_per_video','estimated_vae_decode_tflops_per_video','psnr_rgb_db','ssim_rgb','lpips_alex_v0_1_spatial','yuv_611_db']
 data=[]
 for d in raw:
  d['yuv_611_db']=yr[d['method'],d['setting'],d['sample_id']]['yuv_611_db']
  data.append(dict(method=d['method'],setting=d['setting'],sample_id=d['sample_id'],**{f:float(d[f]) for f in fields}))
 for d in rows(I/'analysis/per_video.csv'):
  data.append(dict(method='Increase',setting='target'+d['target'],sample_id=d['sample_id'],**{f:float(d[f]) for f in fields}))
 assert len(data)==len({(d['method'],d['setting'],d['sample_id']) for d in data})==280
 # Cross-run baseline identity already sealed; assert each new quality pair targets that same native baseline.
 for p in (I/'quality').glob('*/per_video.csv'):
  for q in rows(p):
   sid=q['video_id'].split('_',2)[2]
   base=Path(cfg['reference'])/'baselines'/sid/'video.mp4';assert sid in ids
   assert Path(q['reference']).resolve()==base.resolve();seal(base,q['reference_sha256']);seal(q['candidate'],q['candidate_sha256'])
 summary=[]
 for method,setting in dict.fromkeys((d['method'],d['setting']) for d in data):
  rr=[d for d in data if (d['method'],d['setting'])==(method,setting)];assert len(rr)==20 and {d['sample_id'] for d in rr}==ids
  summary.append(dict(method=method,setting=setting,n=20,speedup=sum(d['baseline_seconds'] for d in rr)/sum(d['generate_seconds'] for d in rr),**{f:st.fmean(d[f] for d in rr) for f in fields}))
 for d in rows(I/'analysis/results.csv'):
  a=next(a for a in summary if a['method']=='Increase' and a['setting']=='target'+d['target'])
  for f in ['speedup',*fields[1:]]:assert abs(a[f]-float(d[f]))<1e-8
 order={'Increase':0,'e391':1,'SeaCache':2,'conservative':3,'aggressive':4};tier={'K23':0,'target1.8':0,'K29':1,'target2.4':1,'K35':2,'target3.0':2}
 summary.sort(key=lambda d:(tier[d['setting']],order[d['method']]))
 delta=[]
 for target,k in [('1.8','23'),('2.4','29'),('3.0','35')]:
  a={d['sample_id']:d for d in data if d['method']=='Increase' and d['setting']=='target'+target}
  for method in ['e391','conservative','aggressive']+(['SeaCache'] if target!='1.8' else []):
   setting='target'+target if method=='SeaCache' else 'K'+k
   b={d['sample_id']:d for d in data if d['method']==method and d['setting']==setting}
   row=dict(increase_target=target,comparator=method,setting=setting,latency_change_pct=100*(sum(x['generate_seconds'] for x in a.values())/sum(x['generate_seconds'] for x in b.values())-1))
   for f in fields[-4:]:
    ds=[a[s][f]-b[s][f] for s in ids];row['delta_'+f]=st.fmean(ds);row['wins_'+f]=sum(x<0 if f=='lpips_alex_v0_1_spatial' else x>0 for x in ds)
   delta.append(row)
 OUT.mkdir(exist_ok=True);write('summary.csv',summary);write('per_video.csv',data);write('increase_paired_deltas.csv',delta)
 text=['# 同VBench20五方法比较','', '全部同20prompt/seed42，20原生baseline复用。Increase60候选与4860对帧已完成，VBench score不计算。RGB/YUV定义与之前一致；同baseline视频SHA及280条视频指标来源核验。','', '|方法|档位|实际速度×|秒/视频|RGB PSNR↑|YUV611↑|SSIM↑|LPIPS↓|','|---|---|---:|---:|---:|---:|---:|---:|']
 for a in summary:text.append(f"|{a['method']}|{a['setting']}|{a['speedup']:.4f}|{a['generate_seconds']:.3f}|{a['psnr_rgb_db']:.4f}|{a['yuv_611_db']:.4f}|{a['ssim_rgb']:.5f}|{a['lpips_alex_v0_1_spatial']:.5f}|")
 text+=['','## Increase减对照（按名义档位对应，须同时看速度差）','', '|目标|对照|耗时变化%|ΔRGB PSNR|ΔYUV611|ΔSSIM|ΔLPIPS|RGB胜出/20|','|---|---|---:|---:|---:|---:|---:|---:|']
 for a in delta:text.append(f"|{a['increase_target']}|{a['comparator']}|{a['latency_change_pct']:+.2f}|{a['delta_psnr_rgb_db']:+.4f}|{a['delta_yuv_611_db']:+.4f}|{a['delta_ssim_rgb']:+.5f}|{a['delta_lpips_alex_v0_1_spatial']:+.5f}|{a['wins_psnr_rgb_db']}|")
 text+=['','## 结论','', '低档Increase实速1.7757×，与三个模型约1.77×接近，四项质量均更好。相对e391 RGB PSNR提高1.4672dB、YUV611提高1.4936dB；相对激进组也有明显改善。','', '中档Increase实速2.3349×，介于K29约2.23×与SeaCache2.4507×之间。质量优于SeaCache，但速度更慢；相比保守/e391更快而质量更低。相较激进组RGB/YUV PSNR略高，但SSIM/LPIPS更差，不存在四指标一致优势。','', '高档Increase实速只有2.6996×，未达到名义3.0×；四项质量虽优于约3×的各对照，但计算更多、耗时更长，不能据此认定同3×速度优势。','', '本次不重新标定或补测。目标由旧mean-threshold映射估算，实际速度偏离必须保留。单seed、20prompt描述性比较，不能作统计显著性或混合训练因果结论。DiT/T5/VAE时间与TFLOPs见summary.csv，YUV为BT.709全范围444、6:1:1通道dB加权。']
 (OUT/'REPORT.md').write_text('\n'.join(text)+'\n');(OUT/'README.md').write_text((HERE/'README.md').read_text());seal(__file__)
 (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',conditions=14,videos=280,same20=True,same_baseline_sha=True,evidence_sha256=S,outputs_sha256={n:sha(OUT/n) for n in ['REPORT.md','summary.csv','per_video.csv','increase_paired_deltas.csv']}),indent=2)+'\n')
 print('\n'.join(text))
if __name__=='__main__':main()
