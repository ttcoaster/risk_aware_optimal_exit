import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.data.observation_scaler import FeatureScaler

class TestObservationScaler(unittest.TestCase):
    def setUp(self):
        self.data=np.array([[0.,1.,2.,3.,4.],[2.,3.,4.,5.,6.],[4.,5.,6.,7.,8.]])
        self.scaler=FeatureScaler.fit(self.data)

    def test_fit_mean(self):
        np.testing.assert_allclose(self.scaler.mean,[2,3,4,5,6])
    def test_fit_scale(self):
        np.testing.assert_allclose(self.scaler.scale,[np.sqrt(8/3)]*5)
    def test_state_unchanged(self):
        raw=np.array([2.,3.,4.,5.,6.,0.6,0.75,0.2])
        np.testing.assert_array_equal(self.scaler.transform(raw)[5:],raw[5:].astype(np.float32))
    def test_standardized_mean_zero(self):
        np.testing.assert_allclose(self.scaler.transform([2,3,4,5,6,1,1,0])[:5],0)
    def test_clipping(self):
        self.assertEqual(self.scaler.transform([1e8]*5+[1,1,0])[0],10)
    def test_constant_feature(self):
        s=FeatureScaler.fit(np.ones((3,5)))
        self.assertEqual(s.scale,(1.,)*5)
    def test_bad_fit(self):
        with self.assertRaises(ValueError): FeatureScaler.fit(np.ones((1,5)))
        with self.assertRaises(ValueError): FeatureScaler.fit(np.ones((3,4)))
    def test_bad_observation(self):
        with self.assertRaises(ValueError): self.scaler.transform([float('nan')]*8)
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'s.json';self.scaler.save(path)
            self.assertEqual(self.scaler,FeatureScaler.load(path))
    def test_schema_guard(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'s.json';path.write_text('{"version": 2, "features": []}')
            with self.assertRaises(ValueError): FeatureScaler.load(path)

if __name__=='__main__':unittest.main()
