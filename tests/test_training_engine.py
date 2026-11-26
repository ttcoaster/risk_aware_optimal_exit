import unittest
from pathlib import Path
from unittest.mock import patch
from src.experiments.configuration import load_plan,canonical_hash
from src.experiments.training_engine import verify_plan,select_jobs,budget
ROOT=Path(__file__).resolve().parents[1]
class TestTrainingEngine(unittest.TestCase):
    def setUp(self):
        self.plan=load_plan(ROOT,'pilot'); self.inputs={'sha256':{'fixture':'hash'},'split_counts':{'train':1310}}
        self.plan.update(status='PLAN_READY',inputs=self.inputs,plan_hash=canonical_hash(dict(config_hash=self.plan['config_hash'],inputs=self.inputs)))
    def verify(self,inputs=None):
        with patch('src.experiments.training_engine.snapshot_inputs',return_value=self.inputs if inputs is None else inputs):
            return verify_plan(ROOT,self.plan)
    def test_valid(self): self.assertEqual(self.verify(),self.inputs)
    def test_input_change(self):
        with self.assertRaises(ValueError): self.verify({})
    def test_job_tamper(self):
        self.plan['jobs'][0]['seed']=99
        with self.assertRaises(ValueError): self.verify()
    def test_hash_tamper(self):
        self.plan['plan_hash']='bad'
        with self.assertRaises(ValueError): self.verify()
    def test_split_tamper(self):
        self.plan['config']['environment']['train_split']='test'
        with self.assertRaises(ValueError): self.verify()
    def test_select(self): self.assertEqual(len(select_jobs(self.plan,['PPO_R1_seed42'])),1)
    def test_duplicate(self):
        with self.assertRaises(ValueError): select_jobs(self.plan,['PPO_R1_seed42']*2)
    def test_unknown(self):
        with self.assertRaises(ValueError): select_jobs(self.plan,['bad'])
    def test_budget(self):
        self.assertEqual(budget(self.plan['jobs'][0],True),2048); self.assertEqual(budget(self.plan['jobs'][0],False),20480)
