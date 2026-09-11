import tempfile
import unittest
from unittest.mock import patch
from dispatch import *

class DispatchTests(unittest.TestCase):
    def test_children_owned_by_executor_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc=Path(tmp)
            for task,children in [('1',''),('2','34'),('3','34 56')]:
                p=proc/'1/task'/task;p.mkdir(parents=True);(p/'children').write_text(children)
            self.assertEqual(child_pids(1,proc),[34,56])

    def test_appended_jobs_seen_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'jobs.json';a=dict(output='a',checkpoint='one');b=dict(output='b',checkpoint='two')
            dump(p,[a]);it=live_jobs(p,2,poll=.001);self.assertEqual(next(it),a)
            dump(p,[b,a]);self.assertEqual(next(it),b)
            with self.assertRaises(StopIteration):next(it)

    def test_reject_job_identity_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'jobs.json';dump(p,[dict(output='a',checkpoint='one')])
            it=live_jobs(p,2,poll=.001);next(it)
            dump(p,[dict(output='a',checkpoint='changed')])
            with self.assertRaises(ValueError):next(it)

    def test_reject_excess_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'jobs.json';dump(p,[dict(output='a'),dict(output='a')])
            with self.assertRaises(ValueError):next(live_jobs(p,2))
            dump(p,[dict(output='a'),dict(output='b')])
            with self.assertRaises(ValueError):next(live_jobs(p,1))

    def test_seven_ready_same_gpu_subset(self):
        c=read(ROOT/'config.json');c['groups']=[g for g in c['groups'] if g['name']!='sea7_a3']
        queues,ready=build_jobs(c)
        self.assertEqual(len(ready),7)
        self.assertEqual([len(queues[g]) for g in range(4)],[63,63,42,42])
        for gpu,jobs in queues.items():
            self.assertTrue(all(j['expected_gpu_uuid']==c['gpu_uuids'][gpu] for j in jobs))
            self.assertTrue(all(j['group']!='sea7_a3' for j in jobs))

if __name__=='__main__':unittest.main()
