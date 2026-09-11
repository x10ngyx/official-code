import sys,os,json,csv,subprocess,fcntl,math,statistics as st,importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];OFFICIAL=PROJECT.parent
sys.path.insert(0,str(HERE.parent/'increase500_iql2_v1'))
from common import PROTOCOL,FIELDS,METRICS,sha256,dump,read,writecsv,REFERENCE
R=Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_increase_vbench20_v1')
def directory(p):
 p.mkdir(parents=True,exist_ok=True)
 if not (p/'README.md').exists():(p/'README.md').write_text('# '+p.name+'\n\nSee experiment root README.\n')
 return p
def link(p,t):
 if p.is_symlink():assert p.resolve()==t.resolve()
 elif p.exists():raise ValueError(p)
 else:p.symlink_to(t)
def prepare():
 directory(R);(R/'README.md').write_text((HERE/'README.md').read_text())
 link(PROJECT/'experiment_results'/R.name,R)
 for n in ['logs','analysis','candidates','quality_inputs','quality','yuv']:directory(R/n)
 if (R/'config.json').exists():return read(R/'config.json')
 previous=R.parent/'ours21_increase500_iql2_v1/config.json';old=read(previous);old['gpu_uuids']={str(i):u for i,u in enumerate(old['gpu_uuids'])}
 mapping=R.parent/'wan21_seacache_speedup_calibration_v1/analysis/speed_threshold_mapping.calibrated.json';m=read(mapping)['mapping']
 assert m['speedups']==[1.5,3.5]
 strategies=[]
 for target in [1.8,2.4,3.0]:
  mean=m['mean_thresholds'][0]+(target-1.5)/2*(m['mean_thresholds'][1]-m['mean_thresholds'][0]);a=mean/3
  strategies.append(dict(target=target,mean_threshold=mean,start=a,end=5*a,threshold_path=[a*(1+4*t/49) for t in range(50)]))
 jobs={str(g):[] for g in range(4)};sources={str(p):sha256(p) for p in [previous,mapping,*HERE.glob('*.py'),OFFICIAL/'SeaCache4Wan21/experiments/linear_increase_matched_v1/schedule.py',OFFICIAL/'SeaCache4Wan21/seacache.py',OFFICIAL/'SeaCache4Wan21/wan21_integration.py',HERE.parent/'increase500_yuv_psnr_v1/run.py']}
 for p in old['prompts']:
  g=next(str(i) for i in range(4) if old['gpu_uuids'][str(i)]==p['baseline_gpu_uuid'])
  base=REFERENCE/'baselines'/p['sample_id'];gen=read(base/'generation.json');assert gen['gpu_uuid']==p['baseline_gpu_uuid'] and gen['protocol']==PROTOCOL
  for n in ['video.mp4','generation.json','measurement.json','COMPLETE.json']:sources[str(base/n)]=sha256(base/n)
  for s in strategies:jobs[g].append(dict(**s,prompt=p,output=str(R/'candidates'/str(s['target'])/p['sample_id'])))
 profile=R.parent/'wan21_seacache_threshold_collection_v1/calflops_profile.json';sources[str(profile)]=sha256(profile)
 cfg=dict(protocol=PROTOCOL,gpu_uuids=old['gpu_uuids'],prompts=old['prompts'],reference=str(REFERENCE),flops_profile=str(profile),strategies=strategies,jobs=jobs,source_sha256=sources,new_calibration=False,vbench_enabled=False)
 assert sum(map(len,jobs.values()))==60
 dump(R/'config.json',cfg);return cfg
def verify(cfg):
 for p,h in cfg['source_sha256'].items():assert sha256(p)==h,p
def command(args,g,name):
 cfg=read(R/'config.json');env=dict(os.environ,CUDA_VISIBLE_DEVICES=cfg['gpu_uuids'][str(g)],CUDA_DEVICE_ORDER='PCI_BUS_ID',TORCH_HOME='/mnt/hdd/xiongyuxiang/tmp/models/torch-cache',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with (R/'logs'/f'{name}.log').open('ab') as f:subprocess.run([sys.executable,*args],cwd=OFFICIAL,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def audit(j,cfg):
 out=Path(j['output']);done=read(out/'COMPLETE.json');assert done['config_sha256']==sha256(R/'config.json')
 for f,h in done['files'].items():assert sha256(out/f)==h
 gen=read(out/'generation.json');assert gen['job']==j and gen['protocol']==PROTOCOL and gen['gpu_uuid']==j['prompt']['baseline_gpu_uuid']
 tr=read(out/'trace.json');t=read(out/'timing.json');assert tr['threshold_path']==j['threshold_path'] and len(tr['decisions'])==len(t['calls'])==100 and t['status']=='success'
 for i,(d,c) in enumerate(zip(tr['decisions'],t['calls'])):
  assert d['step_index']==i//2 and d['branch']==('cond','uncond')[i%2] and d['requested_threshold']==j['threshold_path'][i//2] and d['execution']==d['action']
  assert c['blocks_executed']==(0 if d['action']=='reuse' else 30)
 assert all(d['action']!='reuse' for d in tr['decisions'] if d['step_index'] in [0,49])
 paths=[[d['action']=='reuse' for d in tr['decisions'] if d['branch']==b] for b in ['cond','uncond']];assert paths[0]==paths[1]
 return sum(paths[0])
def key(j):return f"increase_{j['target']}_{j['prompt']['sample_id']}"
def quality(g,cfg):
 jobs=cfg['jobs'][str(g)];inputs=directory(R/'quality_inputs'/f'gpu{g}')
 for side in ['reference','candidate']:directory(inputs/side)
 for j in jobs:
  audit(j,cfg);sid=j['prompt']['sample_id']
  link(inputs/'reference'/(key(j)+'.mp4'),REFERENCE/'baselines'/sid/'video.mp4');link(inputs/'candidate'/(key(j)+'.mp4'),Path(j['output'])/'video.mp4')
 output=R/'quality'/f'gpu{g}'
 if not (output/'summary.json').exists():command([str(OFFICIAL/'VideoMetrics/evaluate.py'),'--reference-dir',str(inputs/'reference'),'--candidate-dir',str(inputs/'candidate'),'--expected-frames','81','--device','cuda:0','--output-dir',str(output)],g,f'quality_gpu{g}')
 assert read(output/'summary.json')['video_count']==len(jobs)
def report(cfg):
 spec=importlib.util.spec_from_file_location('yuv_metric',HERE.parent/'increase500_yuv_psnr_v1/run.py');yuv=importlib.util.module_from_spec(spec);spec.loader.exec_module(yuv);yuv.OUT=R/'yuv';directory(yuv.OUT/'frames');yuv.checks()
 detail=[];yj=[]
 for g,jobs in cfg['jobs'].items():
  qs={q['video_id']:q for q in csv.DictReader((R/'quality'/f'gpu{g}/per_video.csv').open())}
  assert set(qs)=={key(j) for j in jobs}
  for j in jobs:
   skip=audit(j,cfg);sid=j['prompt']['sample_id'];out=Path(j['output']);q=qs[key(j)];c=read(out/'measurement.json');base=read(REFERENCE/'baselines'/sid/'measurement.json')
   assert (int(q['frames']),int(q['width']),int(q['height']))==(81,832,480)
   for side in ['reference','candidate']:assert sha256(q[side])==q[side+'_sha256']
   assert Path(q['candidate']).resolve()==(out/'video.mp4').resolve() and Path(q['reference']).resolve()==(REFERENCE/'baselines'/sid/'video.mp4').resolve()
   assert all(math.isfinite(c[f]) and c[f]>0 for f in FIELDS[:7])
   detail.append(dict(target=j['target'],sample_id=sid,baseline_seconds=base['generate_seconds'],reuse_steps=skip,**{f:c[f] for f in FIELDS[:7]},**{m:float(q[m+'_mean']) for m in METRICS}))
   yj.append(dict(method='increase',setting=str(j['target']),sample_id=sid,reference=q['reference'],candidate=q['candidate'],original_rgb_psnr=q['psnr_rgb_db_mean']))
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(yuv.worker,[j for j in yj if j['sample_id']==p['sample_id']]) for p in cfg['prompts']]
  yr=[r for f in futures for r in f.result()]
 assert len(yr)==len(detail)==60
 writecsv(R/'yuv/per_video.csv',yr)
 for d in detail:
  yy=next(y for y in yr if y['sample_id']==d['sample_id'] and y['setting']==str(d['target']));d['yuv_611_db']=yy['yuv_611_db']
 results=[]
 for s in cfg['strategies']:
  rr=[d for d in detail if d['target']==s['target']];assert len(rr)==20
  results.append(dict(target=s['target'],start=s['start'],end=s['end'],mean_threshold=s['mean_threshold'],speedup=sum(d['baseline_seconds'] for d in rr)/sum(d['generate_seconds'] for d in rr),**{f:st.fmean(d[f] for d in rr) for f in [*FIELDS[:7],*METRICS,'reuse_steps','yuv_611_db']}))
 writecsv(R/'analysis/per_video.csv',detail);writecsv(R/'analysis/results.csv',results)
 text=['# Increase 同VBench20三档测试','', '复用20个原生baseline与同20prompt；无新标定、无VBench score。名义目标由原mean-threshold映射估计，实测速度以表中为准。','', '|目标|阈值起点→终点|实际速度|秒/视频|DiT TFLOPs|RGB PSNR|YUV611 PSNR|SSIM|LPIPS|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for d in results:text.append(f"|{d['target']}|{d['start']:.6f}→{d['end']:.6f}|{d['speedup']:.4f}|{d['generate_seconds']:.3f}|{d['dit_tflops']:.3f}|{d['psnr_rgb_db']:.4f}|{d['yuv_611_db']:.4f}|{d['ssim_rgb']:.5f}|{d['lpips_alex_v0_1_spatial']:.5f}|")
 text+=['','YUV按BT.709全范围444转换后6:1:1通道dB加权，先逐帧后逐视频均值。T5/DiT/VAE分项见CSV；单seed描述性结果。']
 (R/'analysis/REPORT.md').write_text('\n'.join(text)+'\n');verify(cfg)
 evidence={str(p.relative_to(R)):sha256(p) for folder in ['candidates','quality','analysis','yuv'] for p in (R/folder).rglob('*') if p.is_file()}
 dump(R/'VALIDATION.json',dict(status='pass',candidates=60,baseline_reused=20,frame_pairs=4860,cfg_calls=6000,evidence_sha256=evidence))
 dump(R/'COMPLETE.json',dict(status='complete',results_sha256=sha256(R/'analysis/results.csv'),validation_sha256=sha256(R/'VALIDATION.json'),vbench_status='skipped_by_user'))
def main():
 cfg=prepare();verify(cfg)
 if '--prepare' in sys.argv:print(json.dumps(cfg['strategies'],indent=2));return
 with (R/'pipeline.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (R/'COMPLETE.json').exists():return
  try:
   dump(R/'STATUS.json',dict(stage='generation'))
   with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda g:command([str(HERE/'worker.py'),'--root',str(R),'--gpu',str(g)],g,f'worker_gpu{g}'),range(4)))
   dump(R/'STATUS.json',dict(stage='quality'))
   with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda g:quality(g,cfg),range(4)))
   dump(R/'STATUS.json',dict(stage='yuv_and_report'));report(cfg);dump(R/'STATUS.json',dict(stage='complete'))
  except BaseException as e:dump(R/'FAILED.json',dict(error=repr(e)));raise
if __name__=='__main__':main()
