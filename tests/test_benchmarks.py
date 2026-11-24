import unittest
from src.policies.benchmarks import BenchmarkPolicy, BenchmarkConfig

class TestBenchmarks(unittest.TestCase):
    def actions(self, b, prices, horizon=60, config=None):
        p = BenchmarkPolicy(b, config)
        p.reset(prices[0])
        return [p.act(close=x, step=i, horizon=horizon) for i,x in enumerate(prices)]
    def test_hold(self): self.assertEqual(self.actions('B0',[100,80,120]),[0,0,0])
    def test_immediate_retry(self): self.assertEqual(self.actions('B1',[100,100]),[4,4])
    def test_sl(self): self.assertEqual(self.actions('B2',[100,96,95,96]),[0,0,4,0])
    def test_tp(self): self.assertEqual(self.actions('B3',[100,109,110]),[0,0,4])
    def test_combined(self): self.assertEqual(self.actions('B4',[100,95,110]),[0,4,4])
    def test_trailing(self): self.assertEqual(self.actions('B5',[100,120,115,114]),[0,0,0,4])
    def test_schedule(self):
        a=self.actions('B6',[100]*60)
        self.assertEqual([(i,x) for i,x in enumerate(a) if x],[(14,1),(29,2),(44,2),(59,4)])
    def test_random(self):
        a=self.actions('B7',[100]*60)
        self.assertEqual(a,self.actions('B7',[100]*60)); self.assertTrue(set(a)<=set(range(5)))
    def test_reset_peak(self):
        p=BenchmarkPolicy('B5'); p.reset(100); p.act(close=100,step=0,horizon=60); p.act(close=120,step=1,horizon=60)
        p.reset(100); self.assertEqual(p.act(close=100,step=0,horizon=60),0)
    def test_prefix_causality(self):
        for i in range(8): self.assertEqual(self.actions(f'B{i}',[100,105,95]),self.actions(f'B{i}',[100,105,95,200])[:3])
    def test_invalid(self):
        for kwargs in ({'stop_loss':0},{'take_profit':float('nan')},{'interval':0}):
            with self.assertRaises(ValueError): BenchmarkConfig(**kwargs)
        with self.assertRaises(ValueError): BenchmarkPolicy('B8')
        with self.assertRaises(ValueError): BenchmarkPolicy('B0').reset(0)
    def test_sequence(self):
        p=BenchmarkPolicy('B0'); p.reset(100)
        with self.assertRaises(ValueError): p.act(close=100,step=1,horizon=60)
