import unittest
from src.experiments.full_training import full_jobs
class TestFullTraining(unittest.TestCase):
    def setUp(self):
        self.plan={'profile':'full_training','jobs':[dict(algorithm=a,reward=r,seed=s,total_timesteps=102400,checkpoints=[20480,51200,102400]) for a in ('PPO','A2C','DQN') for r in ('R1','R2') for s in (42,7,17,29,53)]}
    def test_grid(self): self.assertEqual(len(full_jobs(self.plan)),30)
    def test_profile(self):
        self.plan['profile']='pilot'
        with self.assertRaises(ValueError): full_jobs(self.plan)
    def test_duplicate(self):
        self.plan['jobs'][0]=self.plan['jobs'][1]
        with self.assertRaises(ValueError): full_jobs(self.plan)
    def test_budget(self):
        self.plan['jobs'][0]['total_timesteps']=20480
        with self.assertRaises(ValueError): full_jobs(self.plan)
