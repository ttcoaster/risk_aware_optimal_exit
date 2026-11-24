import unittest
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
    from src.envs.normalized_observation import NormalizedObservation
    HAS_GYM=True
except ImportError:
    HAS_GYM=False
from src.data.observation_scaler import FeatureScaler

@unittest.skipUnless(HAS_GYM,'gymnasium not installed')
class TestNormalizedObservation(unittest.TestCase):
    def setUp(self):
        class Dummy(gym.Env):
            def __init__(self):
                self.action_space=spaces.Discrete(5)
                self.observation_space=spaces.Box(-np.inf,np.inf,shape=(8,),dtype=np.float32)
            def reset(self,*,seed=None,options=None):
                super().reset(seed=seed)
                return np.array([2,3,4,5,6,.8,.5,.25],dtype=np.float32),{}
            def step(self,action):
                return np.array([2,3,4,5,6,.7,.25,.5],dtype=np.float32),.12,False,True,{'x':1}
        self.env=NormalizedObservation(Dummy(),FeatureScaler((2,3,4,5,6),(1,1,1,1,1)))
    def test_reset(self):
        obs,_=self.env.reset();np.testing.assert_allclose(obs[:5],0)
        self.assertTrue(self.env.observation_space.contains(obs))
    def test_step_preserves_reward_and_info(self):
        self.env.reset();obs,reward,terminated,truncated,info=self.env.step(0)
        self.assertEqual(reward,.12);self.assertFalse(terminated);self.assertTrue(truncated)
        self.assertEqual(info,{'x':1});self.assertAlmostEqual(float(obs[5]),.7,places=5)
    def test_finite_box(self):
        self.assertTrue(np.isfinite(self.env.observation_space.low).all())
        self.assertTrue(np.isfinite(self.env.observation_space.high).all())

if __name__=='__main__':unittest.main()
