import unittest
import pandas as pd
from src.evaluation.behavior_diagnostics import diagnose,POLICIES
class TestBehaviorDiagnostics(unittest.TestCase):
    def setUp(self):
        e=[]; s=[]
        for i in range(376):
            for p in POLICIES:
                k=dict(policy=p,episode_id=f'validation_000270_201901{i:04}',symbol='000270',split='validation')
                e.append(dict(**k,steps=1,remaining_fraction=0,terminal_value=100,terminal_return=0,reward_sum=0,risk_penalty=0,log_return=0,exit_100_step=1,exit_90_step=1,mdd=0,mean_stock_weight=1/60))
                s.append(dict(**k,step=1,shares=0,shares_sold=100,action=4,execution_allowed=True,value=100,reward=0,risk_penalty=0))
        self.e=pd.DataFrame(e); self.s=pd.DataFrame(s)
    def test_grid(self): self.assertEqual(len(diagnose(self.e,self.s)['exit_curves']),840)
    def test_duplicate(self):
        with self.assertRaises(ValueError): diagnose(self.e,pd.concat([self.s,self.s.iloc[:1]]))
    def test_split(self):
        self.s.loc[0,'split']='test'
        with self.assertRaises(ValueError): diagnose(self.e,self.s)
    def test_execution(self):
        self.s.loc[0,'action']=1
        with self.assertRaises(ValueError): diagnose(self.e,self.s)
    def test_reward(self):
        self.s.loc[0,'reward']=1
        with self.assertRaises(ValueError): diagnose(self.e,self.s)
