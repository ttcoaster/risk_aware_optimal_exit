import tempfile, unittest
from pathlib import Path
from src.experiments.checkpoint_tracking import Tracker,schedule

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
