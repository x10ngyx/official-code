"""CPU tests of calibration and full100-pair result construction."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_pair as s


class PairTests(unittest.TestCase):
    def test_real_bracket_no_extrapolation(self):
        source=s.EXP/'wan21_seacache_speedup_calibration_v1/analysis/threshold_summary.csv'
        threshold,bracket=s.interpolate_threshold(s.read_csv(source))
        self.assertAlmostEqual(threshold,.5924018125325419)
        self.assertEqual([float(r['threshold']) for r in bracket],[.5,.6])
        with self.assertRaises(ValueError):s.interpolate_threshold(s.read_csv(source),4.5)

    def test_k38_from_completed_same50(self):
        source=s.EXP/s.prior.NAME/'analysis/results.csv'
        k,fit=s.select_k(s.read_csv(source))
        self.assertEqual(k,38)
        self.assertTrue(37<fit['fractional_k']<38)
        self.assertLess(fit['predicted_speedups']['37'],3.6)
        self.assertGreater(fit['predicted_speedups']['38'],3.6)
        with self.assertRaises(ValueError):s.select_k(s.read_csv(source)[:-1])

    def test_speed_validation_limits(self):
        s.validate_speed(3.60);s.validate_speed(3.63)
        for speed in (3.3,3.9,float('nan')):
            with self.assertRaises(ValueError):s.validate_speed(speed)

    def test_worker_and_runner_parse(self):
        for name in ('run_pair.py','sea_worker.py'):ast.parse((s.HERE/name).read_text())

    def test_formal100_aggregation_real_quality_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'result';previous=Path(tmp)/'previous';root.mkdir();(root/'analysis').mkdir()
            s.dump(root/'config.json',{'synthetic':True});s.dump(root/'calibration/CALIBRATED.json',{'synthetic':True})
            fake=root/'fixture.mp4';fake.write_bytes(b'synthetic, not decoded by test')
            cfg=dict(previous=str(previous),k=38,threshold=.5924,shard_ids={},gpu_uuids={},checkpoint_sha256='policy',baseline_sha256={})
            for g in s.prior.GPUS:
                ids=[f'p{i:03d}' for i in range(g,50,4)];cfg['shard_ids'][str(g)]=ids;cfg['gpu_uuids'][str(g)]=f'GPU-{g}'
                base=[dict(sample_id=sid,**{k:1.0 for k in s.prior.FIELDS[:7]}) for sid in ids]
                for b in base:b['generate_seconds']=252.
                s.dump(previous/'shards'/f'gpu{g}'/'baseline/components.json',{'rows':base})
                for method in s.METHODS:
                    d=root/'shards'/f'gpu{g}'/method;comp=[]
                    s.dump(d/'run.json',dict(protocol=s.prior.PROTOCOL,method='seacache' if method=='seacache' else 'ours',
                        gpu_uuid=f'GPU-{g}',prompts=[dict(sample_id=sid) for sid in ids],threshold=.5924,
                        policy_sha256='policy',state_mode=s.prior.MODE,skip_budget=38))
                    s.dump(d/'COMPLETE.json',dict(videos=len(ids)))
                    for sid in ids:
                        (d/'videos').mkdir(exist_ok=True);(d/'videos'/f'{sid}.mp4').symlink_to(fake)
                        s.dump(d/'timings'/f'{sid}.json',dict(status='success',calls=[{}]*100,pipeline_generate_wall_seconds=70.))
                        s.dump(d/'traces'/f'{sid}.json',dict(threshold=.5924,total_branch_calls=100,step_reuse=38,total_steps=50))
                        comp.append(dict(sample_id=sid,**{k:70. if k=='generate_seconds' else 1.0 for k in s.prior.FIELDS}))
                    s.dump(d/'components.json',dict(rows=comp))
                    qs=[dict(video_id=sid,frames=81,width=832,height=480,reference=str(fake),candidate=str(fake),
                        reference_sha256=s.sha(fake),candidate_sha256=s.sha(fake),
                        psnr_rgb_db_mean=20.,ssim_rgb_mean=.7,lpips_alex_v0_1_spatial_mean=.2) for sid in ids]
                    s.dump(d/'quality/summary.json',dict(video_count=len(ids),frame_count_total=81*len(ids),
                        metrics={m:dict(mean=v) for m,v in zip(s.prior.METRICS,(20.,.7,.2))}))
                    s.prior.write_csv(d/'quality/per_video.csv',qs)
            with patch.object(s,'ROOT',root):s.finalize(cfg)
            result=s.read_csv(root/'analysis/results.csv')
            self.assertEqual(len(result),2)
            self.assertAlmostEqual(float(result[0]['latency_speedup']),3.6)
            self.assertEqual(len(s.read_csv(root/'analysis/per_video.csv')),100)
            self.assertEqual(s.read(root/'analysis/VALIDATION.json')['quality_frames'],8100)
            self.assertEqual(s.read(root/'COMPLETE.json')['vbench_status'],'skipped_by_user')


if __name__=='__main__':unittest.main()
