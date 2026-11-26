import unittest
import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env
from src.envs.accounting import CostConfig
from src.envs.optimal_exit_env import OptimalExitEnv, FEATURES


def sample(n=4, nontradable=()):
    dates = pd.bdate_range('2024-01-02', periods=n)
    d = pd.DataFrame(index=dates)
    d['Open'] = [100., 110., 120., 130.][:n]
    d['Close'] = [100., 115., 125., 135.][:n]
    d['Tradable'] = [i not in nontradable for i in range(n)]
    d['FeatureReady'] = d['Tradable']
    for c in FEATURES: d[c] = np.where(d['Tradable'], 0.01, np.nan)
    return d


class TestOptimalExitEnv(unittest.TestCase):
    def test_check_env(self):
        check_env(OptimalExitEnv(sample(), horizon=3), skip_render_check=True)

    def test_next_open_not_current_close(self):
        e = OptimalExitEnv(sample(), horizon=3, initial_shares=100)
        e.reset()
        obs, r, term, trunc, info = e.step(1)
        self.assertEqual(info['shares_sold'], 25)
        self.assertEqual(info['cash'], 2750.)  # next open 110
        self.assertEqual(info['shares_remaining'], 75)
        self.assertAlmostEqual(info['portfolio_value'], 2750+75*115)
        self.assertEqual(obs.shape, (8,))
        self.assertFalse(term)
        self.assertFalse(trunc)

    def test_no_forced_sale_at_horizon(self):
        e = OptimalExitEnv(sample(), horizon=3, initial_shares=100)
        e.reset()
        for _ in range(3):
            _, _, term, trunc, info = e.step(0)
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(info['shares_remaining'], 100)
        self.assertEqual(info['cash'], 0)
        self.assertEqual(info['terminal_value'], 13500.)
        self.assertFalse(info['forced_sale'])

    def test_full_sale_terminates(self):
        e = OptimalExitEnv(sample(), horizon=3)
        e.reset()
        _, _, term, trunc, info = e.step(4)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(info['shares_remaining'], 0)
        self.assertEqual(info['terminal_value'], info['cash'])

    def test_nontrading_execution_cancelled(self):
        e = OptimalExitEnv(sample(nontradable=(1,)), horizon=3)
        e.reset()
        obs, _, term, trunc, info = e.step(4)
        self.assertFalse(info['execution_allowed'])
        self.assertEqual(info['shares_sold'], 0)
        self.assertEqual(info['shares_remaining'], 100)
        self.assertTrue(np.isfinite(obs).all())
        self.assertFalse(term or trunc)

    def test_costs(self):
        e = OptimalExitEnv(sample(), horizon=3, initial_shares=100,
                           costs=CostConfig(commission_rate=0.001, sell_tax_rate=0.002, slippage_rate=0.01))
        e.reset()
        _, _, _, _, info = e.step(4)
        self.assertAlmostEqual(info['cash'], 100*110*0.99*(1-0.003))


if __name__ == '__main__':
    unittest.main()
