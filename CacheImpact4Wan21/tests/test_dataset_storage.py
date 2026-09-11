import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/single_skip_v1'))
from common import cell_path, sha, write_json
from dataset import initialize_manifest, save_capture, finalize_cell, load_finalized, publish


class DatasetStorageTests(unittest.TestCase):
    def capture(self,step):
        import torch
        from runtime import SingleSkipController
        c=SingleSkipController(step);c.set_scheduler_sigmas(torch.linspace(1,0,51))
        for i in range(50):
            c.observe_latent(torch.full((2,3,4,4),i+.12345),i,torch.tensor([1000.-i]))
            for branch in ['cond','uncond']:
                reuse=c.plan_step(branch=branch,step_index=i,num_steps=50,
                                  feature=torch.full((1,4,2),i+1.),grid_size=torch.tensor([1,2,2]))
                if reuse:c.reuse_residual(branch,i)
                else:c.record_recompute(branch,i,torch.full((1,4,2),i+1.))
        return c

    def fixture(self,root):
        import torch
        rows=[dict(sample_id='p001',prompt='synthetic storage test')]
        write_json(root/'run.json',dict(contract_hash='test',prompts=rows))
        manifests=initialize_manifest(root,rows)
        for step in [0,2]:
            cell=cell_path(root,step,'p001');cell.mkdir(parents=True)
            base=cell_path(root,0,'p001')
            with patch('dataset.LATENT_SHAPE',(2,3,4,4)):
                save_capture(self.capture(step),cell,base,manifests[step,'p001'])
            (cell/'video.mp4').write_bytes(b'synthetic-not-a-real-video')
            (cell/('candidate.mp4' if step else 'baseline.mp4')).symlink_to('video.mp4')
            perf=dict(generate_seconds=10.,estimated_dit_tflops=20.,t5_cuda_seconds=1.,dit_cuda_seconds=8.,
                vae_decode_cuda_seconds=1.,estimated_t5_tflops_per_video=2.,estimated_vae_decode_tflops_per_video=3.,
                terminal_rgb_mse=.001 if step else 0.)
            write_json(cell/'metrics.json',perf)
            write_json(cell/'performance.json',perf)
            write_json(cell/'timing.json',{})
            write_json(cell/'COMPLETE.json',dict(contract_hash='test',files={str(f.relative_to(cell)):sha(f) for f in cell.rglob('*') if f.is_file()}))
            write_json(cell/'quality.json',dict(means={'rgb_mse':perf['terminal_rgb_mse']}))
            (cell/'quality_per_frame.csv').write_text('frame_index,rgb_mse\n0,0\n')
            metric_dir=cell/'video_metrics';metric_dir.mkdir()
            write_json(metric_dir/'summary.json',dict(protocol_id='rgb_full_reference_v1',frame_count_total=81,
                video_count=1,selected_metrics=['psnr','ssim','lpips'],evaluation_elapsed_seconds=1.))
            video=dict(reference=str(base/'video.mp4'),candidate=str(cell/'video.mp4'),
                       reference_sha256=sha(base/'video.mp4'),candidate_sha256=sha(cell/'video.mp4'),frames=81)
            for metric in ['psnr_rgb_db','ssim_rgb','lpips_alex_v0_1_spatial']:
                for stat in ['mean','std_population','min','max']:video[metric+'_'+stat]=1.
            with (metric_dir/'per_video.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(video));w.writeheader();w.writerow(video)
            (metric_dir/'per_frame.csv').write_text('frame_index,psnr_rgb_db,ssim_rgb,lpips_alex_v0_1_spatial\n'+
                ''.join(f'{i},1,1,1\n' for i in range(81)))
            finalize_cell(root,step,rows[0])
        return rows

    def test_full_raw_suffix_and_three_level_snapshot(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=self.fixture(root);cell=cell_path(root,2,'p001')
            self.assertEqual(len(list((cell/'latents').glob('*.pt'))),50)
            last=torch.load(cell/'latents/step_049_input.pt',weights_only=True)
            self.assertEqual(last.dtype,torch.float16)
            self.assertEqual(last.shape,(2,3,4,4))
            self.assertTrue(torch.equal(last,torch.full((2,3,4,4),49.12345).half()))
            completed=load_finalized(root,2,rows[0])
            self.assertEqual(len(completed['step_rows']),50)
            self.assertEqual(len(completed['branch_rows']),100)
            self.assertEqual(sum(r['is_intervention_step'] for r in completed['step_rows']),1)
            self.assertEqual(completed['step_rows'][1]['action'],'reuse')
            self.assertEqual(completed['step_rows'][49]['action'],'recompute')
            self.assertTrue(Path(completed['step_rows'][49]['baseline_latent_path']).exists())
            self.assertIsNotNone(completed['step_rows'][49]['cond_filtered_relative_l1'])
            args=argparse.Namespace(output=root)
            publish(args)
            snapshot=(root/'published/current').resolve()
            summary=json.loads((snapshot/'summary.json').read_text())
            self.assertEqual(summary['published_candidate_count'],1)
            for filename,n in [('trajectory_summary',1),('step_transitions',50),('branch_transitions',100)]:
                self.assertEqual(len((snapshot/'tables'/f'{filename}.jsonl').read_text().splitlines()),n)
                with (snapshot/'tables'/f'{filename}.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),n)
            publish(args)
            self.assertEqual((root/'published/current').resolve(),snapshot)
            (cell/'latents/step_049_input.pt').write_bytes(b'corrupt suffix')
            with self.assertRaises(ValueError):load_finalized(root,2,rows[0])

    def test_manifest_is_step_major_and_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[dict(sample_id='a',prompt='A'),dict(sample_id='b',prompt='B')]
            initialize_manifest(root,rows)
            records=[json.loads(l) for l in (root/'manifests/candidates.jsonl').read_text().splitlines()]
            self.assertEqual([(r['skip_step_1based'],r['sample_id']) for r in records[:4]],[(2,'a'),(2,'b'),(5,'a'),(5,'b')])
            self.assertEqual([r['release_index'] for r in records],list(range(1,99)))
            with self.assertRaises(ValueError):initialize_manifest(root,list(reversed(rows)))

    def test_target_only_capture_cannot_be_saved(self):
        c=self.capture(2);c.input_latents=c.input_latents[:2]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):save_capture(c,Path(d)/'cell',Path(d)/'base',{})


if __name__=='__main__':unittest.main()
