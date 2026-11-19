import unittest

import numpy as np
import pandas as pd

from src.features.build_s1_features import FEATURES, build_features


def make_df(n=40):
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * (1.01 ** np.arange(n))
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * .99,
        "Close": close, "Volume": 1000,
        "Tradable": True,
    }, index=dates)


class TestS1Features(unittest.TestCase):
    def test_returns_and_ready_window(self):
        result = build_features(make_df())
        self.assertEqual(int(result["FeatureReady"].sum()), 20)
        self.assertFalse(result["FeatureReady"].iloc[19])
        self.assertTrue(result["FeatureReady"].iloc[20])
        self.assertAlmostEqual(result["return_1d"].iloc[20], .01)
        self.assertAlmostEqual(result["return_5d"].iloc[20], 1.01**5 - 1)
        self.assertAlmostEqual(result["return_20d"].iloc[20], 1.01**20 - 1)
        self.assertAlmostEqual(result["volatility_20d"].iloc[20], 0.0, places=12)

    def test_nontrading_row_excluded_from_rolling(self):
        df = make_df()
        dt = df.index[25]
        df.loc[dt, "Volume"] = 0
        df.loc[dt, "Tradable"] = False
        result = build_features(df)
        self.assertFalse(result.loc[dt, "FeatureReady"])
        self.assertTrue(result.loc[dt, list(FEATURES)].isna().all())
        self.assertAlmostEqual(result.loc[df.index[26], "return_1d"],
                               df.loc[df.index[26], "Close"] / df.loc[df.index[24], "Close"] - 1)

    def test_no_future_leakage(self):
        df = make_df()
        original = build_features(df)
        altered = df.copy()
        altered.loc[altered.index[30]:, "Close"] *= 10
        new = build_features(altered)
        pd.testing.assert_frame_equal(original.iloc[:30], new.iloc[:30])

    def test_unsorted_rejected(self):
        with self.assertRaises(ValueError):
            build_features(make_df().iloc[::-1])


if __name__ == "__main__":
    unittest.main()
