import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_comparison as suite
import run_4gpu


class Contracts(unittest.TestCase):
    def test_four_gpu_exact_cover(self):
        prompts=[dict(sample_id=str(i),prompt='test') for i in range(5)]
        jobs=run_4gpu.build_jobs(prompts,suite.read(suite.MAPPING))
        self.assertEqual([sum(j['gpu']==g for j in jobs) for g in range(4)],[4,4,4,3])
        self.assertEqual(len({(j['target'],j['sample_id']) for j in jobs}),15)
        for target in suite.TARGETS:
            self.assertEqual({j['sample_id'] for j in jobs if j['target']==target},{str(i) for i in range(5)})

    def test_frozen_calibration(self):
        mapping = suite.read(suite.MAPPING)
        values = [suite.threshold_for(mapping, s) for s in suite.TARGETS]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[0], .1667845774937285)
        self.assertAlmostEqual(values[1], .30261716982044005)
        self.assertAlmostEqual(values[2], .4384497621471516)
        with self.assertRaises(ValueError):
            suite.threshold_for(mapping, 4.)

    def test_real_quality_schema_and_missing_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ids = [str(i) for i in range(5)]
            metrics = ('psnr_rgb_db','ssim_rgb','lpips_alex_v0_1_spatial')
            summary = dict(video_count=5, frame_count_total=405,
                           metrics={key:dict(mean=1.) for key in metrics})
            (root/'summary.json').write_text(json.dumps(summary))
            rows = [dict(video_id=s, frames=81,height=480,width=832,reference='baseline',
                         candidate='candidate',reference_sha256='hash',candidate_sha256='hash',
                         **{key+'_mean':1. for key in metrics}) for s in ids]
            with (root/'per_video.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with patch.object(suite,'sha256',return_value='hash'):
                self.assertEqual(suite.quality_means(root,ids),{key:1. for key in metrics})
                with self.assertRaises(ValueError):
                    suite.quality_means(root,ids[:-1])


if __name__=='__main__':
    unittest.main()
