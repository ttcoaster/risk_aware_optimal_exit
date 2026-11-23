"""STEP 3-3 regression tests against existing accounting and environment."""
import unittest
import numpy as np
import pandas as pd
from src.envs.accounting import CostConfig, SellOnlyAccount
from src.envs.optimal_exit_env import OptimalExitEnv, FEATURES


def frame(opens=(100, 110, 120, 130), closes=(100, 115, 125, 135), suspended=()):
    n = len(opens)
    d = pd.DataFrame(index=pd.bdate_range('2024-01-02', periods=n))
    d['Open'] = list(map(float, opens))
    d['Close'] = list(map(float, closes))
    d['Tradable'] = [i not in suspended for i in range(n)]
    d['FeatureReady'] = d['Tradable']
    for col in FEATURES:
        d[col] = np.where(d['Tradable'], 0.01, np.nan)
    return d


class TestExecutionIntegrity(unittest.TestCase):
    def test_partial_sales_use_remaining_and_round_down(self):
        a = SellOnlyAccount(101)
        z = CostConfig()
        self.assertEqual(a.sell(1, 100, z).shares_sold, 25)
        self.assertEqual(a.sell(2, 100, z).shares_sold, 38)
        self.assertEqual(a.sell(3, 100, z).shares_sold, 28)
        self.assertEqual(a.shares, 10)
        self.assertEqual(a.sell(4, 100, z).shares_sold, 10)
        self.assertEqual(a.shares, 0)
        self.assertAlmostEqual(a.cash, 10100)

    def test_sell_proceeds_cost_components(self):
        a = SellOnlyAccount(100)
        costs = CostConfig(commission_rate=.001, sell_tax_rate=.002, slippage_rate=.01)
        sale = a.sell(2, 110, costs)
        self.assertEqual(sale.shares_sold, 50)
        self.assertAlmostEqual(sale.execution_price, 108.9)
        self.assertAlmostEqual(sale.gross_proceeds, 5445)
        self.assertAlmostEqual(sale.commission, 5.445)
        self.assertAlmostEqual(sale.sell_tax, 10.89)
        self.assertAlmostEqual(sale.net_proceeds, 5428.665)
        self.assertAlmostEqual(a.market_value(115), 5428.665 + 50*115)

    def test_hold_no_fees(self):
        e = OptimalExitEnv(frame(), horizon=3, initial_shares=100,
                           costs=CostConfig(.01, .02, .03))
        e.reset()
        _, _, _, _, info = e.step(0)
        self.assertEqual(info['shares_sold'], 0)
        self.assertEqual(info['cash'], 0)
        self.assertEqual(info['shares_remaining'], 100)
        self.assertAlmostEqual(info['portfolio_value'], 11500)

    def test_next_open_execution_and_mark_next_close(self):
        e = OptimalExitEnv(frame(), horizon=3, initial_shares=100)
        e.reset()
        _, _, _, _, info = e.step(1)
        self.assertEqual(info['shares_sold'], 25)
        self.assertEqual(info['cash'], 2750)
        self.assertEqual(info['portfolio_value'], 2750 + 75*115)

    def test_suspended_next_open_cancels_order(self):
        e = OptimalExitEnv(frame(suspended=(1,)), horizon=3, initial_shares=100)
        e.reset()
        _, _, term, trunc, info = e.step(4)
        self.assertFalse(info['execution_allowed'])
        self.assertEqual(info['shares_sold'], 0)
        self.assertEqual(info['shares_remaining'], 100)
        self.assertFalse(term or trunc)

    def test_terminal_hold_no_liquidation_cost(self):
        e = OptimalExitEnv(frame(), horizon=3, initial_shares=100,
                           costs=CostConfig(.01, .02, .03))
        e.reset()
        for _ in range(3):
            _, _, term, trunc, info = e.step(0)
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(info['shares_remaining'], 100)
        self.assertEqual(info['terminal_value'], 13500)
        self.assertFalse(info['forced_sale'])

    def test_reward_telescopes_to_terminal_log_return(self):
        e = OptimalExitEnv(frame(), horizon=3, initial_shares=100)
        obs, reset_info = e.reset()
        rewards = []
        for action in (1, 2, 0):
            _, r, _, _, info = e.step(action)
            rewards.append(r)
        self.assertAlmostEqual(sum(rewards), np.log(info['terminal_value']/reset_info['initial_value']))

    def test_invalid_cost_rates_rejected(self):
        for kwargs in ({'commission_rate': -0.01}, {'sell_tax_rate': 1.0}, {'slippage_rate': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CostConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
