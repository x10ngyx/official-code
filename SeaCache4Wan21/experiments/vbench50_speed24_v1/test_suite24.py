"""CPU contracts for the fixed-threshold50 supplement."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_suite24 as s


class Tests(unittest.TestCase):
    def test_calibrated_threshold(self):
        mapping=s.read(s.pair.EXP/'wan21_seacache_speedup_calibration_v1/analysis/speed_threshold_mapping.calibrated.json')
        self.assertAlmostEqual(s.threshold_for(mapping),.3026171698,places=9)
        with self.assertRaises(ValueError): s.threshold_for(mapping,4.)
        with self.assertRaises(ValueError): s.threshold_for(dict(mapping,calibration_status='pending'))

    def test_real50_partition(self):
        cfg=s.read(s.pair.ROOT/'config.json')
        self.assertEqual([len(cfg['shard_ids'][str(g)]) for g in s.prior.GPUS],[13,13,12,12])
        self.assertEqual(len(set(sum(cfg['shard_ids'].values(),[]))),50)
        self.assertEqual(len(set(cfg['gpu_uuids'].values())),4)

    def test_sources_parse(self):
        for path in (Path(s.__file__),s.WORKER,Path(__file__).parent/'audit_previous.py'):
            ast.parse(path.read_text())

    def test_full50_finalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'result'; previous=Path(tmp)/'previous'
            (root/'analysis').mkdir(parents=True)
            s.dump(root/'config.json',{'synthetic':True})
            cfg=dict(previous=str(previous),threshold=.3026171698,shard_ids={})
            comp={}; quality={}
            for gpu in s.prior.GPUS:
                ids=[f'p{i}' for i in range(gpu,50,4)]; cfg['shard_ids'][str(gpu)]=ids
                b=[dict(sample_id=sid,**{k:240. if k=='generate_seconds' else 1. for k in s.prior.FIELDS}) for sid in ids]
                c=[dict(sample_id=sid,**{k:100. if k=='generate_seconds' else 1. for k in s.prior.FIELDS}) for sid in ids]
                q=[dict(video_id=sid,psnr_rgb_db_mean=24.,ssim_rgb_mean=.8,lpips_alex_v0_1_spatial_mean=.1) for sid in ids]
                s.dump(previous/'shards'/f'gpu{gpu}'/'baseline/components.json',dict(rows=b))
                d=root/'shards'/f'gpu{gpu}'/'seacache'; comp[gpu]=c; quality[d/'quality']=q
                for name in ('run.json','components.json','COMPLETE.json','quality/summary.json','quality/per_video.csv'):
                    s.dump(d/name,{'synthetic':True})
            with patch.object(s,'ROOT',root),patch.object(s,'validate_sources'),patch.object(s.pair,'check_generation',side_effect=lambda d,g,*a:comp[g]),patch.object(s.prior,'quality_rows',side_effect=lambda p,ids:quality[p]):
                s.finalize(cfg)
            result=s.pair.read_csv(root/'analysis/results.csv')[0]
            self.assertAlmostEqual(float(result['latency_speedup']),2.4)
            self.assertEqual(len(s.pair.read_csv(root/'analysis/per_video.csv')),50)
            self.assertEqual(s.read(root/'analysis/VALIDATION.json')['quality_frames'],4050)
            self.assertEqual(s.read(root/'COMPLETE.json')['vbench_status'],'skipped_by_user')


if __name__=='__main__': unittest.main()
