"""Exercise complete report plumbing using explicitly synthetic, temporary fixtures."""
import tempfile
from unittest.mock import patch
from common import *
import suite
import report
from generate_worker import job_identity
from ours4wan21.online_common import seal

def main():
    root=Path(tempfile.mkdtemp(dir=EXP_ROOT,prefix='iql_report_fixture_'))
    (root/'README.md').write_text('# Synthetic report fixture only\n\nReuses old videos and fabricated Q curves to test report plumbing. NOT experiment evidence.\n')
    for f in ('jobs','evaluation','figures'):(root/f).mkdir()
    uuids=gpu_uuids(['0','1','2','3']);ref=load_evaluation_bundle(REFERENCE)
    prompts=freeze_prompts(ref,uuids)
    groups=[group(f,a) for f in FEATURES for a in LEVELS]
    for g in groups:
        g['training']=g['baseline']
        original=read(Path(g['baseline_analysis'])/'checkpoint_selection.json')
        d=root/g['name'];d.mkdir();g['analysis']=str(d)
        dump(d/'checkpoint_selection.json',original)
        e=d/'extra';e.mkdir();rows=[]
        for method in ('Original','Aggressive'):
            for epoch in range(1,401):
                offset=0 if method=='Original' else int(g['level'][-1])*.2
                rows.append(dict(method=method,epoch=epoch,mean_Q=20+epoch/200+offset,Q_IQR=1+offset+epoch/2000,
                    Q_p25=19,Q_p75=20,states=13738,actions=2))
        writecsv(e/'q_statistics.csv',rows)
        dump(e/'COMPLETE.json',dict(status='complete',epochs_per_method=400,validation_states=13738,
            statistics_sha256=sha256(e/'q_statistics.csv')))
    config=dict(groups=groups,prompts=prompts,gpu_uuids=uuids,source_hashes={},inputs={},
        paths=dict(wan_checkpoint='synthetic-fixture'))
    config['inputs']['flops_profile']=dict(path=str(OLD/'config.json'),sha256=sha256(OLD/'config.json'))
    dump(root/'config.json',config)
    with patch.object(suite,'ROOT',root):queues=suite.evaluation_jobs(config)
    for gpu,jobs in queues.items():
        qs=[]
        for j in jobs:
            old=OLD/'shards'/f'gpu{gpu}'/f"K{j['skip_budget']}";sid=j['sample_id']
            d=Path(j['output']);d.mkdir();identity=job_identity(j,config)
            dump(d/'generation.json',dict(identity=identity,protocol=PROTOCOL,gpu_uuid=j['expected_gpu_uuid']))
            dump(d/'measurement.json',next(r for r in read(old/'components.json')['rows'] if r['sample_id']==sid))
            for name,target in [('video.mp4',old/'videos'/f'{sid}.mp4'),('timing.json',old/'timings'/f'{sid}.json'),
                                ('trace.json',old/'traces'/f'{sid}.json')]:
                (d/name).symlink_to(target)
            seal(d,['generation.json','measurement.json','video.mp4','timing.json','trace.json'],identity=identity)
            q=next(r for r in csv.DictReader((old/'quality/per_video.csv').open()) if r['video_id']==sid)
            q['video_id']=f"{j['group']}_K{j['skip_budget']}_{sid}"
            q['candidate']=str(d/'video.mp4');q['reference']=str(REFERENCE/'baselines'/sid/'video.mp4');qs.append(q)
        qd=root/'evaluation/quality'/f'gpu{gpu}';qd.mkdir()
        writecsv(qd/'per_video.csv',qs)
        dump(qd/'summary.json',dict(video_count=len(jobs),frame_count_total=81*len(jobs)))
    with patch.object(report,'ROOT',root),patch.object(report,'verify_training',lambda g:None):report.main()
    assert read(root/'COMPLETE.json')['candidates']==240
    assert len(list(csv.DictReader((root/'results.csv').open())))==24
    print('PASS temporary synthetic 240-video / 4000-Q-row report fixture: '+str(root),flush=True)

if __name__=='__main__':main()
