"""STEP 3-6: integrated environment checks; no existing source modifications."""
import math
import unittest
import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env
from src.envs.accounting import CostConfig
from src.envs.optimal_exit_env import OptimalExitEnv, FEATURES
from src.envs.rewards import RewardConfig


def frame(closes=(100, 95, 90, 110, 105), nontradable=()):
    n = len(closes)
    f = pd.DataFrame(index=pd.bdate_range('2024-01-02', periods=n))
    f['Close'] = np.asarray(closes, dtype=float)
    f['Open'] = np.asarray(closes, dtype=float)
    f['Tradable'] = [i not in nontradable for i in range(n)]
    f['FeatureReady'] = f['Tradable']
    for col in FEATURES:
        f[col] = np.where(f['Tradable'], 0.01, np.nan)
    return f


def run(f, actions, mode='R1', lam=0.5, costs=None):
    e = OptimalExitEnv(f, horizon=len(f)-1, initial_shares=100,
                       costs=costs or CostConfig(),
                       reward_config=RewardConfig(mode=mode, drawdown_lambda=lam))
    obs, meta = e.reset(seed=7)
    rewards, infos = [], []
    for action in actions:
        obs, reward, terminated, truncated, info = e.step(action)
        assert np.isfinite(obs).all()
        rewards.append(float(reward)); infos.append(info)
        if terminated or truncated:
            break
    return e, meta, rewards, infos, terminated, truncated


class TestEnvIntegration(unittest.TestCase):
    def test_gymnasium_api(self):
        check_env(OptimalExitEnv(frame(), horizon=4), skip_render_check=True)

    def test_r1_telescoping_realized_value(self):
        e, meta, rewards, infos, term, trunc = run(frame(), [1, 0, 2, 0])
        self.assertFalse(term); self.assertTrue(trunc)
        self.assertAlmostEqual(sum(rewards), math.log(infos[-1]['terminal_value']/meta['initial_value']), places=10)
        self.assertAlmostEqual(infos[-1]['cash']+infos[-1]['shares_remaining']*105, infos[-1]['terminal_value'])
        self.assertFalse(infos[-1]['forced_sale'])

    def test_r2_same_accounting_different_reward(self):
        f=frame()
        a=run(f,[0,0,0,0],mode='R1')
        b=run(f,[0,0,0,0],mode='R2',lam=0.5)
        self.assertEqual(a[3][-1]['terminal_value'],b[3][-1]['terminal_value'])
        self.assertEqual(a[3][-1]['shares_remaining'],b[3][-1]['shares_remaining'])
        self.assertLess(sum(b[2]),sum(a[2]))
        self.assertTrue(any(x['risk_penalty']>0 for x in b[3]))

    def test_r2_lambda_zero_equals_r1(self):
        a=run(frame(),[0,0,0,0],mode='R1')
        b=run(frame(),[0,0,0,0],mode='R2',lam=0)
        np.testing.assert_allclose(a[2],b[2],atol=1e-12)

    def test_early_full_sale(self):
        e,meta,rewards,infos,term,trunc=run(frame(),[4])
        self.assertTrue(term);self.assertFalse(trunc)
        self.assertEqual(infos[-1]['shares_remaining'],0)
        self.assertEqual(infos[-1]['terminal_value'],infos[-1]['cash'])

    def test_suspension_and_terminal_mark(self):
        e,meta,rewards,infos,term,trunc=run(frame(nontradable=(1,2)),[4,4,0,0])
        self.assertEqual(infos[0]['shares_sold'],0)
        self.assertEqual(infos[1]['shares_sold'],0)
        self.assertEqual(infos[-1]['shares_remaining'],100)
        self.assertTrue(trunc)
        self.assertFalse(infos[-1]['forced_sale'])

    def test_transaction_cost_decreases_terminal_value(self):
        a=run(frame(),[1,0,0,0])
        b=run(frame(),[1,0,0,0],costs=CostConfig(commission_rate=0.001,sell_tax_rate=0.002,slippage_rate=0.001))
        self.assertLess(b[3][-1]['terminal_value'],a[3][-1]['terminal_value'])

    def test_sb3_check_env(self):
        try:
            from stable_baselines3.common.env_checker import check_env as sb3_check
        except ImportError:
            self.skipTest('stable-baselines3 not installed')
        sb3_check(OptimalExitEnv(frame(),horizon=4),warn=True)

if __name__=='__main__':
    unittest.main()
