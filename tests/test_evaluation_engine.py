import unittest
from src.evaluation.engine import episode_seed,summarize
class TestEvaluationEngine(unittest.TestCase):
    def row(self,p,e,r): return dict(policy=p,episode_id=e,terminal_return=r,mdd=.1,mean_stock_weight=.5)
    def test_seed(self):
        self.assertEqual(episode_seed('a'),episode_seed('a')); self.assertNotEqual(episode_seed('a'),episode_seed('b'))
    def test_pair(self):
        s,p=summarize([self.row('B0','a',.1),self.row('PPO_R1','a',.2)])
        self.assertAlmostEqual(p[1]['excess_return'],.1); self.assertEqual(s[1]['n'],1)
    def test_duplicate(self):
        with self.assertRaises(ValueError): summarize([self.row('B0','a',0)]*2)
    def test_unmatched(self):
        with self.assertRaises(ValueError): summarize([self.row('B0','a',0),self.row('B1','b',0)])
    def test_missing_baseline(self):
        with self.assertRaises(ValueError): summarize([self.row('B1','a',0)])
