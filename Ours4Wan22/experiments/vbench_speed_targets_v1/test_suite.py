"""CPU suite contracts and completed-stage recovery without GPU generation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('vbench_suite',Path(__file__).with_name('run.py'))
suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)


class SuiteTests(unittest.TestCase):
    def test_video_metrics_schema_and_coverage(self):
        quality=dict(video_count=5,frame_count_total=225,metrics={
            'psnr_rgb_db':{'mean':25.},'ssim_rgb':{'mean':.8},'lpips_alex_v0_1_spatial':{'mean':.1}})
        self.assertEqual(suite.quality_values(quality,5),dict(psnr_rgb_db=25.,ssim=.8,lpips=.1))
        with self.assertRaises(ValueError):suite.quality_values(quality,200)

    def test_model_registry_symlink_but_not_arbitrary_external_path(self):
        from ours4wan22 import shared
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);models=root/'models';models.mkdir();external=root/'disk';external.mkdir()
            (models/'weights').symlink_to(external,target_is_directory=True)
            with patch.object(shared,'MODELS',models):
                self.assertEqual(shared.under(models/'weights',models),external)
                with self.assertRaises(ValueError):shared.under(external,models)
                with self.assertRaises(ValueError):shared.under(models/'../disk',models)

    def test_default_subset_and_full_set(self):
        self.assertEqual([r['sample_id'] for r in suite.jobs_for()],suite.DEFAULT_IDS)
        full=suite.jobs_for(all_prompts=True)
        suite.validate_vbench_jobs(full,'vbench200')
        full[0]['prompt']='mismatched actual generation text'
        with self.assertRaises(ValueError):suite.validate_vbench_jobs(full,'vbench200')
        with self.assertRaises(ValueError):suite.validate_vbench_jobs(suite.jobs_for(),'vbench200')
        suite.validate_vbench_jobs(suite.jobs_for(),'custom')

    def test_invalid_prompt_selection(self):
        for ids in (['vbench200_001']*2,['bad'],[]):
            with self.assertRaises(ValueError):suite.jobs_for(ids)

    def test_commands_use_new_calibration_and_one_baseline(self):
        args=suite.parser().parse_args(['--policy',str(suite.MODELS/'fixture.pt'),
            '--profile','/fixture/profile.json','--wan22-root','/fixture/source',
            '--gpu','1','--output','/all/yiran07-disk3/huteng_data/exp/test_suite_contract'])
        with patch.object(suite,'sha256',return_value='fixture'),patch.object(suite,'implementation_hashes',return_value={}):
            plan=suite.build_plan(args)
        self.assertEqual([t['K'] for t in plan['targets']],[24,31,36])
        self.assertEqual(len(plan['stages']),7)
        self.assertEqual(sum(s['name']=='baseline' for s in plan['stages']),1)
        self.assertEqual(plan['vbench_mode'],'custom')
        for s in plan['stages']:
            if 'target' in s:self.assertIn('--calibration',s['command'])
            if s['kind']=='evaluate':self.assertIn('--vbench-mode',s['command'])
        args.targets=[2.,2.]
        with self.assertRaises(ValueError):suite.build_plan(args)

    def test_dry_run_does_not_launch_or_create_outputs(self):
        with patch.object(suite,'build_plan',return_value={'stages':[]}),patch.object(suite.subprocess,'run') as launch,patch.object(suite,'result_dir') as mkdir,patch('builtins.print'):
            suite.run(type('Args',(),{'dry_run':True})())
            launch.assert_not_called();mkdir.assert_not_called()

    def test_sealed_stage_reused_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'candidate';out.mkdir()
            plan={'output':str(root)};stage={'name':'target','kind':'generate','output':str(out),'target':2.,'K':27}
            for name in ('manifest.json','run.json','performance.json'):(out/name).write_text('{}')
            suite.write(root/'stages/target.json',dict(files={n:suite.sha256(out/n) for n in ('manifest.json','run.json','performance.json')}))
            with patch.object(suite,'validate_generation') as validate,patch.object(suite.subprocess,'run') as launch:
                suite.execute_stage(stage,plan,{},'GPU-fixture')
                validate.assert_called_once();launch.assert_not_called()
                (out/'performance.json').write_text('{"changed":true}')
                with self.assertRaises(ValueError):suite.execute_stage(stage,plan,{},'GPU-fixture')

    def test_unsealed_stage_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'partial';out.mkdir()
            with patch.object(suite.subprocess,'run') as launch:
                with self.assertRaises(ValueError):suite.execute_stage({'output':str(out),'name':'partial'}, {'output':str(root)}, {},'GPU-fixture')
                launch.assert_not_called()


if __name__=='__main__':unittest.main()
