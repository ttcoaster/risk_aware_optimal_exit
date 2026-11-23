"""STEP 3-5: edge cases against the unchanged STEP 3-4 environment."""
import unittest
import numpy as np
import pandas as pd
from src.envs.optimal_exit_env import OptimalExitEnv

FEATURES = ('return_1d','return_5d','return_20d','volatility_20d','ma_gap_20d')


def make_frame(*, opens=(100, 110, 120, 130), closes=(100, 115, 125, 135), tradable=(True, True, True, True)):
    dates = pd.bdate_range('2023-01-02', periods=len(opens))
    data = {'Open': list(opens), 'Close': list(closes), 'Tradable': list(tradable),
            'FeatureReady': list(tradable), 'Volume': [1000 if t else 0 for t in tradable]}
    data.update({name: [0.0 if t else np.nan for t in tradable] for name in FEATURES})
    return pd.DataFrame(data, index=dates)


class TestMarketEdgeCases(unittest.TestCase):
    def test_suspended_execution_cancels_sale(self):
        env = OptimalExitEnv(make_frame(tradable=(True, False, True, True)), horizon=3, initial_shares=100)
        env.reset()
        _, _, term, trunc, info = env.step(4)
        self.assertFalse(info['execution_allowed'])
        self.assertEqual(info['shares_sold'], 0)
        self.assertEqual(info['shares_remaining'], 100)
        self.assertFalse(term or trunc)

    def test_cancelled_order_not_carried_forward(self):
        env = OptimalExitEnv(make_frame(tradable=(True, False, True, True)), horizon=3, initial_shares=100)
        env.reset()
        env.step(4)
        _, _, _, _, info = env.step(0)
        self.assertEqual(info['shares_remaining'], 100)
        self.assertEqual(info['shares_sold'], 0)

    def test_consecutive_suspension_still_ends_at_horizon(self):
        env = OptimalExitEnv(make_frame(tradable=(True, False, False, False)), horizon=3, initial_shares=100)
        env.reset()
        for i in range(3):
            obs, _, term, trunc, info = env.step(4)
            self.assertEqual(info['shares_remaining'], 100)
            self.assertTrue(np.isfinite(obs).all())
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertFalse(info['forced_sale'])
        self.assertAlmostEqual(info['terminal_value'], 13500)

    def test_final_day_nontradable_no_forced_sale(self):
        env = OptimalExitEnv(make_frame(tradable=(True, True, True, False)), horizon=3, initial_shares=100)
        env.reset()
        env.step(0)
        env.step(0)
        _, _, term, trunc, info = env.step(4)
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(info['shares_remaining'], 100)
        self.assertEqual(info['shares_sold'], 0)
        self.assertFalse(info['forced_sale'])

    def test_valid_execution_after_suspension(self):
        env = OptimalExitEnv(make_frame(tradable=(True, False, True, True)), horizon=3, initial_shares=100)
        env.reset()
        env.step(4)
        _, _, term, _, info = env.step(4)
        self.assertTrue(term)
        self.assertEqual(info['shares_sold'], 100)
        self.assertEqual(info['shares_remaining'], 0)
        self.assertAlmostEqual(info['cash'], 12000)

    def test_invalid_close_rejected(self):
        frame = make_frame()
        frame.loc[frame.index[-1], 'Close'] = np.nan
        with self.assertRaises(ValueError):
            OptimalExitEnv(frame, horizon=3)

    def test_invalid_ready_feature_rejected(self):
        frame = make_frame()
        frame.loc[frame.index[1], FEATURES[0]] = np.inf
        with self.assertRaises(ValueError):
            OptimalExitEnv(frame, horizon=3)

    def test_suspended_feature_is_not_future_filled(self):
        frame = make_frame(tradable=(True, False, True, True))
        frame.loc[frame.index[0], FEATURES[0]] = 0.12
        frame.loc[frame.index[2], FEATURES[0]] = -0.08
        env = OptimalExitEnv(frame, horizon=3)
        obs0, _ = env.reset()
        obs1, _, _, _, _ = env.step(0)
        obs2, _, _, _, _ = env.step(0)
        self.assertAlmostEqual(float(obs0[0]), 0.12, places=6)
        self.assertAlmostEqual(float(obs1[0]), 0.12, places=6)
        self.assertAlmostEqual(float(obs2[0]), -0.08, places=6)


if __name__ == '__main__':
    unittest.main()
