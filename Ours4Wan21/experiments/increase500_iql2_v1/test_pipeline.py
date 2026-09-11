import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pipeline as p

class PipelineTests(unittest.TestCase):
    def test_profiles(self):
        groups=p.groups()
        a,b=[g['training_config'] for g in groups]
        self.assertEqual({k for k in a if a[k]!=b[k]},{'tau','beta','weight_max'})
        self.assertEqual((a['tau'],a['beta'],a['weight_max']),(.6,1,20))
        self.assertEqual((b['tau'],b['beta'],b['weight_max']),(.9,3,100))
        self.assertTrue(all(g['mode']==p.MODE and g['training_config']['epochs']==400 for g in groups))
        self.assertEqual(groups[0]['cache'],groups[1]['cache'])
        # create_result registers each leaf name globally in experiment_results.
        names=[Path(g[k]).name for g in groups for k in ('training','analysis')]
        names += [Path(groups[0]['cache']).name,p.ROOT.name+'_features']
        self.assertEqual(len(names),len(set(names)))
        self.assertTrue(all(name.startswith(p.ROOT.name) for name in names))

    def test_exact_reference_jobs(self):
        ref=p.read(p.REFERENCE/'manifest.json')
        with tempfile.TemporaryDirectory() as d,patch.object(p,'ROOT',Path(d)):
            (p.ROOT/'jobs').mkdir()
            groups=p.groups()
            for g in groups:
                dest=Path(g['analysis']);dest.mkdir(parents=True)
                p.dump(dest/'checkpoint_selection.json',dict(checkpoint='synthetic.pt',checkpoint_sha256='synthetic',checkpoint_epoch=350))
            cfg=dict(groups=groups,prompts=ref['rows'],gpu_uuids=sorted({r['baseline_gpu_uuid'] for r in ref['rows']}))
            queues=p.evaluation_jobs(cfg);jobs=sum(queues.values(),[])
            self.assertEqual(len(jobs),120)
            for g in groups:
                for k in p.BUDGETS:
                    selected=[j for j in jobs if j['group']==g['name'] and j['skip_budget']==k]
                    self.assertEqual({j['sample_id'] for j in selected},{r['sample_id'] for r in ref['rows']})
                    self.assertEqual(len(selected),20)
            for gpu,js in queues.items():
                self.assertTrue(all(j['expected_gpu_uuid']==cfg['gpu_uuids'][gpu] for j in js))

    def test_stage_dependency_and_no_vbench(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'suite';root.mkdir();exp=Path(d)
            cfg=dict(groups=[{'name':'conservative'},{'name':'aggressive'}])
            p.dump(root/'config.json',cfg);events=[]
            def event(name):
                return lambda *args:events.append(name)
            with patch.object(p,'ROOT',root),patch.object(p,'EXP_ROOT',exp),patch.object(p,'COLLECTION',exp/'pending'),\
                 patch.object(p,'verify_sources'),patch.object(p,'dataset_ready',side_effect=[False,True]),\
                 patch.object(p.time,'sleep',side_effect=event('wait')),\
                 patch.object(p,'features',side_effect=event('features')),patch.object(p,'cache',side_effect=event('cache')),\
                 patch.object(p,'train_group',side_effect=event('train_select')),\
                 patch.object(p,'evaluation_jobs',return_value={i:[] for i in range(4)}),\
                 patch.object(p,'execute',side_effect=event('generate')),\
                 patch.object(p,'quality',side_effect=event('quality')),patch.object(p,'report',side_effect=event('report')):
                p.run_pipeline()
            self.assertEqual(events,['wait','features','cache']+['train_select']*2+['generate']*4+['quality']*4+['report'])
            self.assertEqual(p.read(root/'STATUS.json')['stage'],'complete')

if __name__=='__main__':unittest.main()
