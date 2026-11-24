import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env
from src.envs.training_env import TrainingEnv
from src.envs.optimal_exit_env import OptimalExitEnv
from src.envs.rewards import RewardConfig
from src.data.observation_scaler import FeatureScaler, FEATURES

class Loader:
    def __init__(self):
        self.rs = {s+str(i):SimpleNamespace(episode_id=s+str(i),symbol='005930',split=s) for s in ('train','validation','test') for i in range(3)}
    def records(self,s): return tuple(r for r in self.rs.values() if r.split==s)
    def record(self,e): return self.rs[e]
    def make_env(self,e,*,expected_split,**kw):
        if self.rs[e].split != expected_split: raise ValueError('split violation')
        f=pd.DataFrame({'Open':[100.,90.,95.], 'Close':[100.,90.,95.], 'Tradable':True,'FeatureReady':True},index=pd.date_range('2020-01-01',periods=3))
        for k in FEATURES: f[k]=0.
        return OptimalExitEnv(f,horizon=2,**kw)

class TestTrainingEnv(unittest.TestCase):
    def make(self,mode='R1',split='train'):
        return TrainingEnv(Loader(),FeatureScaler((0.,)*5,(1.,)*5),split=split,reward_config=RewardConfig(mode=mode))
    def test_seed(self):
        a,b=self.make(),self.make()
        a.reset(seed=7); b.reset(seed=7)
        self.assertEqual([a.reset()[1]['episode_id'] for _ in range(20)],[b.reset()[1]['episode_id'] for _ in range(20)])
    def test_split_guard(self):
        a=self.make()
        with self.assertRaises(ValueError): a.reset(options={'episode_id':'test0'})
    def test_fixed_episode(self):
        a=self.make(split='validation'); self.assertEqual(a.reset(options={'episode_id':'validation1'})[1]['episode_id'],'validation1')
    def test_drawdown(self):
        a=self.make('R2'); o,_=a.reset(); self.assertEqual(o.shape,(9,)); self.assertEqual(o[-1],0)
        o,r,_,_,info=a.step(0); self.assertAlmostEqual(float(o[-1]),.1); self.assertAlmostEqual(r,np.log(.9)-.05)
        o,*_=a.step(0); self.assertAlmostEqual(float(o[-1]),.05)
    def test_reward_parity(self):
        a,b=self.make(),self.make('R2'); a.reset(seed=1); b.reset(seed=1)
        x,y=a.step(1),b.step(1); np.testing.assert_array_equal(x[0],y[0]); self.assertEqual(x[4]['portfolio_value'],y[4]['portfolio_value']); self.assertNotEqual(x[1],y[1])
    def test_reset_account(self):
        a=self.make(); a.reset(); a.step(4); o,info=a.reset(); self.assertEqual(o[6],1); self.assertEqual(o[8],0); self.assertEqual(info['initial_value'],10000)
    def test_done_guard(self):
        a=self.make(); a.reset(); self.assertTrue(a.step(4)[2]); self.assertRaises(RuntimeError,a.step,0)
    def test_horizon(self):
        a=self.make(); a.reset(); a.step(0); x=a.step(0); self.assertTrue(x[3]); self.assertFalse(x[4]['forced_sale'])
    def test_gymnasium(self): check_env(self.make(),skip_render_check=True)
    def test_sb3(self):
        from stable_baselines3.common.env_checker import check_env as sb3_check
        sb3_check(self.make())
