import unittest
import torch
from schedule import ScheduledController,SeaCacheController,SeaCacheConfig,ENDPOINTS,linear_path
from run_experiment import pair

class ScheduleTests(unittest.TestCase):
    def drive(self,controller):
        controller.set_scheduler_sigmas(torch.linspace(.99,.01,50))
        for step in range(50):
            for branch,scale in [('cond',1.),('uncond',2.)]:
                feature=torch.arange(1,9,dtype=torch.float32).reshape(1,8,1)*(1+step*.035)*scale
                reuse=controller.plan_step(branch=branch,step_index=step,num_steps=50,feature=feature,grid_size=torch.tensor([2,2,2]))
                if reuse:controller.reuse_residual(branch,step)
                else:controller.record_recompute(branch,step,feature)
        return controller.summary()
    def test_constant_path_exact_original_parity(self):
        a=self.drive(SeaCacheController(SeaCacheConfig(.24)));b=self.drive(ScheduledController([.24]*50))
        self.assertEqual(a['reuse'],b['reuse'])
        for x,y in zip(a['decisions'],b['decisions']):
            self.assertEqual(x,{k:v for k,v in y.items() if k!='requested_threshold'})
    def test_increase_index_and_boundaries(self):
        for start,end in ENDPOINTS:
            path=linear_path(start,end);trace=self.drive(ScheduledController(path))
            self.assertAlmostEqual(path[0],start);self.assertAlmostEqual(path[-1],end)
            self.assertTrue(all(a<b for a,b in zip(path,path[1:])))
            for row in trace['decisions']:
                self.assertEqual(row['requested_threshold'],path[row['step_index']])
                if row['step_index'] in (0,49):self.assertEqual(row['action'],'recompute')
    def test_calibration_interpolation_and_no_extrapolation(self):
        rows=[dict(speedup=1.5,threshold=.1),dict(speedup=2.5,threshold=.3)]
        self.assertAlmostEqual(pair.interpolate_threshold(rows,2.)[0],.2)
        with self.assertRaises(ValueError):pair.interpolate_threshold(rows,3.)
    def test_bad_paths(self):
        for path in ([.1]*49,[float('nan')]*50,[-1.]*50):
            with self.assertRaises(ValueError):ScheduledController(path)
if __name__=='__main__':unittest.main()
