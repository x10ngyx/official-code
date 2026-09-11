import unittest
import numpy as np
from analyze import selection

class SelectionTests(unittest.TestCase):
    def test_all_candidates_including_endpoints(self):
        q=np.tile(np.arange(8).reshape(1,4,2),(21,1,1)).astype(float)
        a=np.zeros((21,4),dtype=int)
        result=selection(list(range(180,201)),q,a)
        self.assertEqual([r['epoch'] for r in result['ranked']],list(range(180,201)))
        self.assertEqual(result['ranked'][0]['neighbor_count'],1)
        self.assertEqual(result['ranked'][-1]['neighbor_count'],1)
        self.assertTrue(all(r['neighbor_count']==2 for r in result['ranked'][1:-1]))
        self.assertEqual(result['selected']['epoch'],200)
    def test_bad_shapes_rejected(self):
        with self.assertRaises(ValueError):selection(list(range(181,201)),np.zeros((20,4,2)),np.zeros((20,4)))
    def test_zero_iqr_guard(self):
        result=selection(list(range(180,201)),np.zeros((21,4,2)),np.zeros((21,4)))
        self.assertTrue(all(np.isfinite(r['mean_normalized_dq']) for r in result['ranked']))
    def test_actor_gate_excludes_unstable_tail(self):
        q=np.tile(np.arange(200).reshape(1,100,2),(21,1,1)).astype(float)
        a=np.zeros((21,100),dtype=int);a[-1]=1
        result=selection(list(range(180,201)),q,a)
        self.assertEqual(result['selected']['epoch'],198)
        self.assertEqual(result['gate'],.96)

if __name__=='__main__':unittest.main()
