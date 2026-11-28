import unittest
from src.experiments.pilot import pilot_jobs
class TestPilot(unittest.TestCase):
    def plan(self):
        return dict(profile='pilot',jobs=[dict(algorithm=a,reward=r,seed=42,total_timesteps=20480,checkpoints=[2048,10240,20480]) for a in ('PPO','A2C','DQN') for r in ('R1','R2')])
    def test_grid(self): self.assertEqual(len(pilot_jobs(self.plan())),6)
    def test_profile(self):
        p=self.plan(); p['profile']='full_training'
        with self.assertRaises(ValueError): pilot_jobs(p)
    def test_duplicate(self):
        p=self.plan(); p['jobs'][0]=p['jobs'][1]
        with self.assertRaises(ValueError): pilot_jobs(p)
    def test_budget(self):
        p=self.plan(); p['jobs'][0]['total_timesteps']=2048
        with self.assertRaises(ValueError): pilot_jobs(p)
