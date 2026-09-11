import copy
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter
import run
from ours4wan21.selection import selected_paths


class IncreaseTests(unittest.TestCase):
    def test_selection(self):
        source = run.rows(run.SOURCE/'manifests/random_runnable.jsonl')
        prompts = list({r['sample_id']:r for r in source if r['prompt_rank']<1000}.values())
        plan, probes = run.sample_plan(prompts)
        self.assertEqual(run.sample_plan(prompts),(plan,probes))
        self.assertEqual([r['release_index'] for r in plan],list(range(500)))
        self.assertEqual(len({r['sample_id'] for r in plan}),500)
        self.assertEqual(Counter(r['split'] for r in plan),{'train':400,'val':50,'test':50})
        self.assertEqual(Counter(r['shard_index'] for r in plan),{0:125,1:125,2:125,3:125})
        self.assertTrue(all(r['split']=='train' for r in probes))
        self.assertFalse({r['sample_id'] for r in probes}&{r['sample_id'] for r in plan})
        bins=Counter(min(4,int((r['target_speedup']-1.5)/.4)) for r in plan)
        self.assertEqual(bins,{i:100 for i in range(5)})
        original={r['sample_id']:r['split'] for r in prompts}
        self.assertTrue(all(original[r['sample_id']]==r['split'] for r in plan))

    def test_calibration_bounds(self):
        points=[dict(scale=.02,speedup=1.4),dict(scale=.3,speedup=3.8)]
        self.assertAlmostEqual(run.interpolate(points,2.6),.16)
        for target in (1.,4.):
            with self.assertRaises(ValueError):run.interpolate(points,target)
        with self.assertRaises(ValueError):
            run.interpolate([dict(scale=.1,speedup=2),dict(scale=.2,speedup=1.8)],1.9)
        p=run.path_for(.2)
        self.assertEqual(len(p),50)
        self.assertEqual((p[0],p[-1]),(.2,1.))
        self.assertTrue(all(a<b for a,b in zip(p,p[1:])))

    def test_mixed_selection_and_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'completed').mkdir()
            (root/'manifests').mkdir()
            records=[]
            for sid in range(1000):
                split='train' if sid<800 else 'val' if sid<900 else 'test'
                for c in range(3):
                    records.append(dict(trajectory_id=f'{sid}_r{c}',sample_id=str(sid),split=split,
                        policy_family='random_continuous_seacache_threshold',protocol=run.PROTOCOL))
            for sid in list(range(400))+list(range(800,850))+list(range(900,950)):
                records.append(dict(trajectory_id=f'{sid}_i',sample_id=str(sid),
                    split='train' if sid<800 else 'val' if sid<900 else 'test',
                    policy_family=run.FAMILY,protocol=run.PROTOCOL))
            def seal():
                selected=[]
                for r in records:
                    path=root/'completed'/f"{r['trajectory_id']}.json"
                    path.write_text(json.dumps({'trajectory_row':r}))
                    selected.append({**r,'complete_path':str(path.relative_to(root)),'complete_sha256':run.sha(path)})
                manifest=root/'manifests/mixed_runnable.jsonl'
                manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
                selection=root/'selection.json'
                selection.write_text(json.dumps(dict(schema='ours4wan21_mixed_subset_v1',selected_count=3500,
                    rows=selected,source_manifest_sha256=run.sha(manifest),
                    family_counts=dict(Counter(r['policy_family'] for r in records)),
                    split_counts=dict(Counter(r['split'] for r in records)))))
                return selection
            self.assertEqual(len(selected_paths(seal(),root)),3500)
            records[-1]['split']='train'
            with self.assertRaisesRegex(ValueError,'leakage'):selected_paths(seal(),root)
            records[-1]['split']='test'
            records[-1]['policy_family']='random_continuous_seacache_threshold'
            with self.assertRaisesRegex(ValueError,'contract mismatch'):selected_paths(seal(),root)


if __name__=='__main__':unittest.main()
