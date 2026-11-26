import copy
import tempfile
import unittest
from pathlib import Path
from src.experiments.configuration import load_plan,validate,canonical_hash,read_yaml,digest,snapshot_inputs
ROOT=Path(__file__).resolve().parents[1]
class TestExperimentConfiguration(unittest.TestCase):
    def setUp(self): self.plan=load_plan(ROOT,'pilot'); self.c=copy.deepcopy(self.plan['config'])
    def check(self): validate(self.c['environment'],self.c['experiment'],self.c['algorithms'])
    def test_pilot(self): self.assertEqual(len(self.plan['jobs']),6)
    def test_full(self): self.assertEqual(len(load_plan(ROOT,'full_training')['jobs']),30)
    def test_split(self):
        self.c['environment']['train_split']='test'
        with self.assertRaises(ValueError): self.check()
    def test_duplicate_seed(self):
        self.c['experiment']['seeds']=[42,42]
        with self.assertRaises(ValueError): self.check()
    def test_alignment(self):
        self.c['experiment']['checkpoints']=[2049,20480]
        with self.assertRaises(ValueError): self.check()
    def test_parameter_guard(self):
        self.c['algorithms']['PPO']['gamma']=.99
        with self.assertRaises(ValueError): self.check()
    def test_hash(self):
        self.assertEqual(canonical_hash({'a':1,'b':2}),canonical_hash({'b':2,'a':1})); self.assertNotEqual(canonical_hash({'a':1}),canonical_hash({'a':2}))
    def test_duplicate_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.yaml'; p.write_text('a: 1\na: 2')
            with self.assertRaises(ValueError): read_yaml(p)
    def test_missing_input(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError): snapshot_inputs(d,self.c['environment'])
    def test_file_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x'; p.write_bytes(b'a'); h=digest(p); p.write_bytes(b'b'); self.assertNotEqual(h,digest(p))
