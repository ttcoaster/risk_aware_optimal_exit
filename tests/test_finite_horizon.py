import unittest
import gymnasium as gym
import numpy as np
from src.envs.finite_horizon import FiniteHorizon
from stable_baselines3.common.vec_env import DummyVecEnv

class Stub(gym.Env):
    observation_space=gym.spaces.Box(0,1,(1,),dtype=np.float32)
    action_space=gym.spaces.Discrete(2)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); return np.zeros(1,dtype=np.float32),{}
    def step(self,a):
        return np.ones(1,dtype=np.float32),.1,a==1,a==0,{'terminal_value':110.,'forced_sale':False}

class TestFiniteHorizon(unittest.TestCase):
    def test_horizon(self):
        e=FiniteHorizon(Stub()); e.reset(); o,r,t,tr,i=e.step(0)
        self.assertTrue(t); self.assertFalse(tr); self.assertEqual(r,.1); self.assertEqual(i['terminal_value'],110.)
        self.assertFalse(i['forced_sale']); self.assertTrue(i['horizon_reached'])
    def test_early_exit(self):
        e=FiniteHorizon(Stub()); e.reset(); _,_,t,tr,i=e.step(1)
        self.assertTrue(t); self.assertFalse(tr); self.assertNotIn('horizon_reached',i)
    def test_no_timeout_bootstrap(self):
        e=DummyVecEnv([lambda:FiniteHorizon(Stub())]); e.reset()
        _,_,d,i=e.step(np.array([0])); self.assertTrue(d[0]); self.assertFalse(i[0]['TimeLimit.truncated']); np.testing.assert_array_equal(i[0]['terminal_observation'],[1]); e.close()
