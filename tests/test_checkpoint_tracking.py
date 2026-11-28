import tempfile, unittest
from pathlib import Path
from src.experiments.checkpoint_tracking import Tracker,schedule
from src.experiments.checkpoint_tracking import (
    preserve_rng_state
)
class TestCheckpointTracking(unittest.TestCase):
    def test_schedule(self):
        self.assertEqual(schedule({'checkpoints':[2048,10240,20480]},False),[2048,10240,20480])
        self.assertEqual(schedule({},True),[1024,2048])
    def test_duplicate(self):
        with self.assertRaises(ValueError): Tracker('.', [1024,1024], 'h','j')
    def test_missed(self):
        t=Tracker('.', [1024,2048],'h','j')
        with self.assertRaisesRegex(ValueError,'missed'): t.capture(type('M',(),{'num_timesteps':1025})(),1025,1)
    def test_transition(self):
        t=Tracker('.', [1024,2048],'h','j')
        with self.assertRaisesRegex(ValueError,'transition'): t.capture(type('M',(),{'num_timesteps':1024})(),1023,1)
    def test_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'incomplete'): Tracker(d,[1024,2048],'h','j').finish()
    def test_before_target(self):
        t=Tracker('.', [1024,2048],'h','j'); t.capture(type('M',(),{'num_timesteps':0})(),0,0); self.assertEqual(t.rows,[])

    def test_rng_preservation(self):
        import random
        import numpy as np
        import torch

        with preserve_rng_state():
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

            with preserve_rng_state():
                expected = (
                    random.random(),
                    float(np.random.random()),
                    float(torch.rand(1).item())
                )

            with preserve_rng_state():
                random.seed(999)
                np.random.seed(999)
                torch.manual_seed(999)

                random.random()
                np.random.random()
                torch.rand(100)

            actual = (
                random.random(),
                float(np.random.random()),
                float(torch.rand(1).item())
            )

            self.assertEqual(expected, actual)

    def test_rng_exception(self):
        import random
        import numpy as np
        import torch

        with preserve_rng_state():
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

            with preserve_rng_state():
                expected = (
                    random.random(),
                    float(np.random.random()),
                    float(torch.rand(1).item())
                )

            with self.assertRaises(RuntimeError):
                with preserve_rng_state():
                    random.random()
                    np.random.random()
                    torch.rand(100)

                    raise RuntimeError("test error")

            actual = (
                random.random(),
                float(np.random.random()),
                float(torch.rand(1).item())
            )

            self.assertEqual(expected, actual)
