import math
import unittest
from src.envs.rewards import RewardConfig, compute_reward

class TestRewards(unittest.TestCase):
    def test_r1_log_return(self):
        r = compute_reward(100, 95, 100, RewardConfig())
        self.assertAlmostEqual(r.reward, math.log(.95))
        self.assertEqual(r.penalty, 0)

    def test_r2_drawdown_increase(self):
        r = compute_reward(100, 95, 100, RewardConfig('R2', .5))
        self.assertAlmostEqual(r.reward, math.log(.95) - .025)
        self.assertAlmostEqual(r.drawdown_increase, .05)

    def test_recovery_no_penalty(self):
        r = compute_reward(95, 98, 100, RewardConfig('R2', .5))
        self.assertEqual(r.penalty, 0)

    def test_new_peak_no_penalty(self):
        r = compute_reward(100, 110, 100, RewardConfig('R2', .5))
        self.assertEqual(r.peak_value, 110)
        self.assertEqual(r.penalty, 0)

    def test_no_repeated_penalty_for_flat_underwater(self):
        r = compute_reward(95, 95, 100, RewardConfig('R2', .5))
        self.assertEqual(r.penalty, 0)

    def test_zero_lambda_equals_r1(self):
        a = compute_reward(100, 92, 100, RewardConfig('R1'))
        b = compute_reward(100, 92, 100, RewardConfig('R2', 0))
        self.assertEqual(a.reward, b.reward)

    def test_invalid_config(self):
        for cfg in [('R3', 0.5), ('R2', -1), ('R2', float('nan'))]:
            with self.assertRaises(ValueError): RewardConfig(*cfg)

    def test_invalid_values(self):
        with self.assertRaises(ValueError): compute_reward(0, 90, 100, RewardConfig())
        with self.assertRaises(ValueError): compute_reward(100, 90, 95, RewardConfig())

if __name__ == '__main__': unittest.main()
