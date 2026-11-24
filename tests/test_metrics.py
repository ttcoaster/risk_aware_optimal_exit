import unittest
from src.evaluation.metrics import StepRecord as R, episode_metrics as E, distribution_metrics as D, paired_excess
class TestMetrics(unittest.TestCase):
    def test_hold(self):
        m=E(100,100,[R(110,100,0,0,True,0),R(99,100,0,0,True,0)],2)
        self.assertAlmostEqual(m['mdd'],.1); self.assertAlmostEqual(m['terminal_return'],-.01); self.assertEqual(m['mean_stock_weight'],1)
    def test_padding(self):
        m=E(100,100,[R(90,0,90,4,True,1)],60)
        self.assertAlmostEqual(m['mean_stock_weight'],1/60); self.assertEqual(m['exit_100_step'],1); self.assertAlmostEqual(m['downside_deviation'],.1/60**.5)
    def test_partial(self):
        m=E(100,100,[R(100,75,25,1,True,0),R(100,38,62,2,True,0)],2)
        self.assertEqual(m['exit_50_step'],2); self.assertIsNone(m['exit_90_step']); self.assertEqual(m['fills'],2)
    def test_cancel(self):
        m=E(100,100,[R(100,100,0,4,False,0)],1)
        self.assertEqual(m['cancelled_orders'],1); self.assertEqual(m['noop_rate'],1)
    def test_rounding(self):
        m=E(100,1,[R(100,1,0,1,True,0)],1); self.assertEqual(m['noop_rate'],1)
    def test_flat(self):
        m=E(100,100,[R(100,100,0,0,True,0)],1); self.assertIsNone(m['daily_sharpe']); self.assertIsNone(m['daily_sortino']); self.assertIsNone(m['noop_rate'])
    def test_incomplete(self):
        with self.assertRaises(ValueError): E(100,100,[R(100,100,0,0,True,0)],2)
    def test_bad_sale(self):
        with self.assertRaises(ValueError): E(100,100,[R(100,99,1,0,True,0)],1)
    def test_cost(self):
        m=E(100,100,[R(99,0,99,4,True,1)],1); self.assertEqual(m['cost_fraction'],.01)
    def test_tail(self):
        m=D([-.2,-.1,.1,.2],.375); self.assertAlmostEqual(m['var_loss'],.1); self.assertAlmostEqual(m['cvar_loss'],1/6)
    def test_ceq(self):
        self.assertAlmostEqual(D([-.1,.1])['ceq'],-.015)
    def test_pair(self):
        self.assertAlmostEqual(paired_excess({'a':.2},{'a':.1})['a'],.1)
        with self.assertRaises(ValueError): paired_excess({'a':.2},{'b':.1})
    def test_invalid_distribution(self):
        with self.assertRaises(ValueError): D([float('nan')])
