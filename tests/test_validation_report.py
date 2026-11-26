import unittest
from src.evaluation.validation_report import diagnostics
P=('B0','PPO_R1','PPO_R2','A2C_R1','A2C_R2','DQN_R1','DQN_R2')
def rows():
    return [dict(policy=p,episode_id=e,symbol='005930',split='validation',terminal_return=.1 if e=='a' else -.1,mdd=.1,mean_stock_weight=.5,exit_100_step=None,exit_90_step=None,remaining_fraction=.1,sell_requests=2,fills=1,cost_fraction=.001) for p in P for e in ('a','b')]
class TestValidationReport(unittest.TestCase):
    def test_grid(self):
        b,p,m=diagnostics(rows(),{'a','b'},P); self.assertEqual((len(b),len(p),len(m)),(7,6,14)); self.assertEqual(b[0]['noop_rate'],.5); self.assertEqual(p[0]['delta_return'],0)
    def test_missing(self):
        with self.assertRaises(ValueError): diagnostics(rows()[:-1],{'a','b'},P)
    def test_duplicate(self):
        r=rows(); self.assertRaises(ValueError,diagnostics,r+r[:1],{'a','b'},P)
    def test_split(self):
        r=rows(); r[0]['split']='test'; self.assertRaises(ValueError,diagnostics,r,{'a','b'},P)
    def test_no_requests(self):
        r=rows(); r[0]['sell_requests']=r[0]['fills']=r[1]['sell_requests']=r[1]['fills']=0; self.assertIsNone(diagnostics(r,{'a','b'},P)[0][0]['noop_rate'])
