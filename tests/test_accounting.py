import unittest
from src.envs.accounting import SellOnlyAccount, CostConfig


class TestAccounting(unittest.TestCase):
    def test_partial_sales_use_remaining_shares(self):
        a = SellOnlyAccount(100)
        self.assertEqual(a.sell(1, 10000, CostConfig()).shares_sold, 25)
        self.assertEqual(a.sell(2, 10000, CostConfig()).shares_sold, 37)
        self.assertEqual(a.shares, 38)
        self.assertEqual(a.cash, 620000)

    def test_terminal_mark_to_market_no_forced_sale(self):
        a = SellOnlyAccount(100)
        a.sell(1, 10000, CostConfig())
        self.assertEqual(a.terminal_value(11000), 1075000)
        self.assertEqual(a.shares, 75)

    def test_costs_and_slippage(self):
        a = SellOnlyAccount(10)
        result = a.sell(4, 10000, CostConfig(.001, .002, .01))
        self.assertAlmostEqual(result.execution_price, 9900)
        self.assertAlmostEqual(result.net_proceeds, 98703)
        self.assertEqual(a.shares, 0)
        self.assertAlmostEqual(a.cash, 98703)

    def test_zero_sale_when_too_few_shares(self):
        a = SellOnlyAccount(1)
        self.assertEqual(a.sell(1, 10000, CostConfig()).shares_sold, 0)
        self.assertEqual(a.shares, 1)
        self.assertEqual(a.sell(4, 10000, CostConfig()).shares_sold, 1)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): SellOnlyAccount(0)
        a = SellOnlyAccount(10)
        with self.assertRaises(ValueError): a.sell(5, 100, CostConfig())
        with self.assertRaises(ValueError): a.market_value(-1)
        with self.assertRaises(ValueError): CostConfig(sell_tax_rate=-0.1)


if __name__ == '__main__':
    unittest.main()
