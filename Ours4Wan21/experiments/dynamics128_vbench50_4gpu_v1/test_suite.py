"""CPU-only contracts; synthetic data retains production *_mean CSV schema."""
import csv
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_suite as suite


def source_rows():
    return [dict(sample_id=f'p{i:03d}',prompt_en=f'Prompt {i}') for i in range(200)]


def measured(sid,seconds,baseline=False):
    fields=suite.FIELDS[:7] if baseline else suite.FIELDS
    r=dict(sample_id=sid,**{k:1.0 for k in fields})
    r['generate_seconds']=seconds
    return r


class SuiteTests(unittest.TestCase):
    def test_random_selection_single_draw_and_partition(self):
        rows=source_rows();chosen=suite.select_prompts(rows)
        self.assertEqual(chosen,sorted(random.Random(42).sample(rows,50),key=lambda r:r['sample_id']))
        self.assertEqual(chosen,suite.select_prompts(rows))
        shards=suite.partition(chosen)
        self.assertEqual([len(shards[g]) for g in suite.GPUS],[13,13,12,12])
        self.assertEqual({r['sample_id'] for rs in shards.values() for r in rs},{r['sample_id'] for r in chosen})
        with self.assertRaises(ValueError):suite.select_prompts(rows[:-1]+[rows[0]])
        with self.assertRaises(ValueError):suite.partition(chosen[:-1]+[chosen[0]])

    def test_real_prompt_metadata_16_dimensions(self):
        root=suite.OFFICIAL/'Vbench200'
        chosen=suite.select_prompts([json.loads(x) for x in (root/'prompts.jsonl').read_text().splitlines()])
        full=json.loads((root/'VBench200_full_info.json').read_text());prompts={r['prompt_en'] for r in chosen}
        self.assertEqual({d for r in full if r['prompt_en'] in prompts for d in r['dimension']},
                         set(suite.read_json(suite.OFFICIAL/'VbenchEvaluation/dimensions.json')['dimensions']))
        self.assertEqual({r['sample_id'] for r in chosen}&{f'vbench200_{s}' for s in ('001','016','056','135','159')},{'vbench200_056'})

    def test_native_missing_predictor_fields_and_ratio(self):
        base=[measured(f'p{i}',200+i,True) for i in range(50)]
        cand=[measured(f'p{i}',(200+i)/2) for i in range(50)]
        result=suite.summarize(base,cand)
        self.assertEqual(result['latency_speedup'],2)
        self.assertEqual(result['baseline_predictor_tflops_mean'],0)
        with self.assertRaises(ValueError):suite.summarize(base,cand[:-1])
        with self.assertRaises(ValueError):suite.summarize(base,cand[:-1]+[cand[0]])
        cand[0]['generate_seconds']=float('nan')
        with self.assertRaises(ValueError):suite.summarize(base,cand)

    def test_real_quality_schema(self):
        root=suite.EXP_ROOT/'ours21_random3000_12groups_v1_vbench5_speed_targets_v1/candidates'/suite.MODE/'K23/quality/video_metrics'
        ids=[f'vbench200_{s}' for s in ('001','016','056','135','159')]
        rows=suite.quality_rows(root,ids)
        self.assertEqual(len(rows),5)
        self.assertIn('psnr_rgb_db_mean',rows[0])

    def test_full_150_pair_analysis_and_vbench_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'analysis').mkdir();suite.dump(root/'config.json',{'synthetic':True})
            fake=root/'video.mp4';fake.write_bytes(b'synthetic fixture, not actual video')
            ids=[r['sample_id'] for r in suite.select_prompts(source_rows())]
            config={'shard_ids':{str(g):ids[g::4] for g in suite.GPUS}}
            def folder(g,label):return root/'shards'/f'gpu{g}'/label
            for g in suite.GPUS:
                shard=config['shard_ids'][str(g)]
                for k in suite.KS:
                    d=folder(g,f'K{k}')
                    suite.dump(d/'performance.json',dict(baseline=[measured(s,200,True) for s in shard],candidate=[measured(s,100) for s in shard]))
                    suite.dump(d/'quality/summary.json',dict(video_count=len(shard),frame_count_total=len(shard)*81,
                        metrics={m:dict(mean=v) for m,v in zip(suite.METRICS,(25,.8,.1))}))
                    qs=[dict(video_id=s,frames=81,height=480,width=832,reference=str(fake),candidate=str(fake),
                        reference_sha256=suite.sha256(fake),candidate_sha256=suite.sha256(fake),
                        **{m+'_mean':v for m,v in zip(suite.METRICS,(25,.8,.1))}) for s in shard]
                    suite.write_csv(d/'quality/per_video.csv',qs)
            suite.analyze(root,config,folder,False)
            self.assertFalse(suite.read_json(root/'analysis/QUALITY_VALIDATION.json')['vbench_complete'])
            for label in suite.LABELS:
                suite.dump(root/'vbench'/label/'aggregate.json',dict(aggregate_scores=dict(total_score=.8,quality_score=.8,semantic_score=.8)))
            suite.analyze(root,config,folder,True)
            with (root/'analysis/results.csv').open() as stream:result=list(csv.DictReader(stream))
            self.assertEqual(len(result),3)
            self.assertEqual(float(result[0]['psnr_rgb_db']),25)
            self.assertEqual(float(result[0]['vbench50_total_score']),.8)
            with (root/'analysis/per_video.csv').open() as stream:self.assertEqual(len(list(csv.DictReader(stream))),150)
            self.assertEqual(suite.read_json(root/'analysis/VALIDATION.json')['quality_frames'],12150)
            suite.dump(root/'VBENCH_SKIPPED_BY_USER.json',dict(status='skipped_by_user',video_metrics_enabled=True))
            self.assertFalse(suite.vbench_enabled(root))
            suite.analyze(root,config,folder,False,final=True)
            validation=suite.read_json(root/'analysis/VALIDATION.json')
            self.assertEqual(validation['vbench_status'],'skipped_by_user')
            self.assertFalse(validation['vbench_complete'])
            self.assertNotIn('vbench50_total_score',(root/'analysis/results.csv').read_text())
            self.assertIn('按用户要求跳过',(root/'analysis/RESULTS.md').read_text())
            # Duplicate ID must fail even if video_count/frame_count remain correct.
            d=folder(0,'K23')/'quality'
            with (d/'per_video.csv').open() as stream:bad=list(csv.DictReader(stream))
            bad[-1]=bad[0]
            suite.write_csv(d/'per_video.csv',bad)
            with self.assertRaises(ValueError):suite.quality_rows(d,config['shard_ids']['0'])

    def test_scope_override_only_allows_approved_runner_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);script=str(Path(suite.__file__))
            original=dict(protocol='fixed',source_sha256={script:'old','model':'unchanged'})
            suite.dump(root/'config.json',original)
            suite.dump(root/'VBENCH_SKIPPED_BY_USER.json',dict(status='skipped_by_user',video_metrics_enabled=True,
                previous_runner_sha256='old',updated_runner_sha256='new'))
            updated=dict(protocol='fixed',source_sha256={script:'new','model':'unchanged'})
            suite.validate_resume_config(root,updated)
            with self.assertRaises(ValueError):suite.validate_resume_config(root,{**updated,'protocol':'changed'})
            with self.assertRaises(ValueError):suite.validate_resume_config(root,{**updated,'source_sha256':{script:'new','model':'changed'}})


if __name__=='__main__':unittest.main()
