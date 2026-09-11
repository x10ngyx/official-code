import csv,json,hashlib,sys,math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1];sys.path.insert(0,str(PROJECT.parent/'VideoMetrics'))
from video_metrics.video import decode_video_rgb
from video_metrics.core import psnr_per_frame
R=Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_increase500_iql2_v1');OUT=R/'analysis/yuv_psnr20'
COLS=['y_psnr','u_psnr','v_psnr','yuv_611_db','yuv_weighted_mse_db','rgb_psnr']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(Path(p).open()))
def write(p,rr):
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def db(mse):return np.where(mse<1e-10,100.,-10*np.log10(np.maximum(mse,1e-300)))
def channel_mses(diff):
 y=.2126*diff[0]+.7152*diff[1]+.0722*diff[2]
 u=(diff[2]-y)/1.8556;v=(diff[0]-y)/1.5748
 return np.array([np.mean(c*c) for c in [y,u,v]])
def checks():
 assert np.all(db(np.zeros(3))==100)
 diff=np.ones((3,2,3))*.1;m=channel_mses(diff)
 assert np.allclose(m,[.01,0,0],atol=1e-15)
 assert abs(np.dot(db(m),[6,1,1])/8-40)<1e-10
 # Independent explicit transform of random images vs linear difference shortcut.
 rng=np.random.default_rng(42);a=rng.random((3,7,9));b=rng.random((3,7,9))
 mat=np.array([[.2126,.7152,.0722],[-.2126/1.8556,-.7152/1.8556,.5],[.5,-.7152/1.5748,-.0722/1.5748]])
 x=np.einsum('ij,jhw->ihw',mat,a);z=np.einsum('ij,jhw->ihw',mat,b)
 assert np.allclose(channel_mses(a-b),((x-z)**2).mean((1,2)),rtol=1e-12)
def worker(jobs):
 base=decode_video_rgb(jobs[0]['reference']);assert base.shape==(81,3,480,832)
 results=[]
 for j in jobs:
  candidate=decode_video_rgb(j['candidate']);assert candidate.shape==base.shape
  rgb=psnr_per_frame(base,candidate);frames=[]
  for i in range(81):
   m=channel_mses(base[i].astype(np.float64)-candidate[i].astype(np.float64));p=db(m)
   frames.append(dict(frame=i,y_psnr=p[0],u_psnr=p[1],v_psnr=p[2],yuv_611_db=float(np.dot(p,[6,1,1])/8),yuv_weighted_mse_db=float(db(np.dot(m,[6,1,1])/8)),rgb_psnr=rgb[i],capped_channels=int(sum(m<1e-10))))
  d={**j,**{k:float(np.mean([f[k] for f in frames])) for k in COLS},'capped_channels':sum(f['capped_channels'] for f in frames)}
  assert abs(d['rgb_psnr']-float(j['original_rgb_psnr']))<1e-8,(j,d['rgb_psnr'])
  write(OUT/'frames'/f"{j['method']}_{j['setting']}_{j['sample_id']}.csv",frames)
  results.append(d)
 return results
def main():
 checks();OUT.mkdir(parents=True,exist_ok=True);(OUT/'frames').mkdir(exist_ok=True)
 (OUT/'README.md').write_text((HERE/'README.md').read_text());(OUT/'frames/README.md').write_text('# Per-frame metrics\n\n81 aligned frames per candidate; frame index starts at0.\n')
 prev=R/'analysis/final_comparison20';v=read(prev/'VALIDATION.json')
 for n,h in v['output_sha256'].items():assert sha(prev/n)==h
 old=rows(prev/'per_video.csv');assert len(old)==220
 cfg=read(R/'config.json');ref=Path(cfg['reference']);jobs=[]
 archive=R.parent/'ours21_dynamics128_e391_vbench50_random42_4gpu_v1'
 old_index={(r['sample_id'],r['k']):r for r in rows(archive/'analysis/per_video.csv')}
 for d in old:
  method,setting,sid=d['method'],d['setting'],d['sample_id']
  if method in ['conservative','aggressive']:p=R/f'evaluation/candidates/{method}/{setting}/{sid}/video.mp4'
  else:
   gpu=old_index[sid,'23']['gpu']
   folder=archive/f'shards/gpu{gpu}/{setting}' if method=='e391' else R.parent/('seacache_wan21_vbench50_speed'+('24' if setting=='target2.4' else '30')+'_4gpu_v1')/f'shards/gpu{gpu}/seacache'
   q=next(q for q in rows(folder/'quality/per_video.csv') if q['video_id']==sid);p=Path(q['candidate'])
  bp=ref/'baselines'/sid/'video.mp4'
  for f in [p,bp]:
   expected=v['evidence_sha256'].get(str(f))
   if expected is None:expected=next(h for n,h in v['evidence_sha256'].items() if Path(n).resolve()==f.resolve())
   assert sha(f)==expected
  jobs.append(dict(method=method,setting=setting,sample_id=sid,reference=str(bp),candidate=str(p),reference_sha256=sha(bp),candidate_sha256=sha(p),original_rgb_psnr=d['psnr_rgb_db']))
 (OUT/'manifest.json').write_text(json.dumps(jobs,indent=2)+'\n')
 result=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(worker,[j for j in jobs if j['sample_id']==sid]) for sid in sorted({j['sample_id'] for j in jobs})]
  for f in as_completed(futures):result.extend(f.result());print(f'{len(result)}/220 videos complete',flush=True)
 result.sort(key=lambda d:(d['method'],d['setting'],d['sample_id']));assert len(result)==220
 write(OUT/'per_video.csv',result);agg=[]
 speed={(d['method'],d['setting']):d['speedup'] for d in rows(prev/'summary.csv')}
 for method,setting in dict.fromkeys((d['method'],d['setting']) for d in result):
  rr=[d for d in result if (d['method'],d['setting'])==(method,setting)];assert len(rr)==20
  agg.append(dict(method=method,setting=setting,n=20,speedup=speed[method,setting],**{k:float(np.mean([d[k] for d in rr])) for k in COLS}))
 write(OUT/'summary.csv',agg);paired=[]
 for group in ['conservative','aggressive']:
  for k in [23,29,35]:
   for comparator,setting in [('e391','K'+str(k))]+([('SeaCache','target'+{29:'2.4',35:'3.0'}[k])] if k!=23 else []):
    a={d['sample_id']:d for d in result if d['method']==group and d['setting']=='K'+str(k)};b={d['sample_id']:d for d in result if d['method']==comparator and d['setting']==setting}
    ds=[a[s]['yuv_611_db']-b[s]['yuv_611_db'] for s in a]
    paired.append(dict(group=group,k=k,comparator=comparator,setting=setting,delta_yuv_611_db=float(np.mean(ds)),wins=sum(x>0 for x in ds),n=20))
 write(OUT/'paired_deltas.csv',paired)
 text=['# 同20prompt YUV加权PSNR','', '口径：BT.709 full-range YCbCr 4:4:4，从既有RGB解码转换；6:1:1加权各通道dB，逐帧后逐视频等权。峰值1，MSE<1e-10记100dB。不是原始YUV420编码平面PSNR，也不是先加权MSE再转dB。','', '|方法|档位|实速|RGB PSNR|Y PSNR|YUV 6:1:1 PSNR|','|---|---|---:|---:|---:|---:|']
 for d in agg:text.append(f"|{d['method']}|{d['setting']}|{float(d['speedup']):.4f}|{d['rgb_psnr']:.4f}|{d['y_psnr']:.4f}|{d['yuv_611_db']:.4f}|")
 text+=['','|新组|K|对照|Δ加权PSNR|胜出/20|','|---|---|---|---:|---:|']
 for d in paired:text.append(f"|{d['group']}|{d['k']}|{d['comparator']} {d['setting']}|{d['delta_yuv_611_db']:+.4f}|{d['wins']}|")
 text+=['','220条既有候选、20条原生baseline全部复用；17820对帧，无重生成。RGB PSNR全部与原指标核对。中档SeaCache实速不同，不能作等速优势结论；SeaCache低档缺失。单seed描述性比较，改变指标不会改变视频本身。','', (HERE/'README.md').read_text()]
 (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
 assert all(sha(j[side])==j[side+'_sha256'] for j in jobs for side in ['reference','candidate'])
 outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='VALIDATION.json'}
 (OUT/'VALIDATION.json').write_text(json.dumps(dict(status='pass',videos=220,paired_frames=17820,rgb_reproduction_max_abs_error=max(abs(d['rgb_psnr']-float(d['original_rgb_psnr'])) for d in result),analytic_tests='pass',capped_channels=sum(d['capped_channels'] for d in result),source_validation_sha256=sha(prev/'VALIDATION.json'),script_sha256=sha(__file__),output_sha256=outputs),indent=2)+'\n')
 print('\n'.join(text))
if __name__=='__main__':main()
