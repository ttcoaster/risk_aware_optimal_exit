import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import pandas as pd
import numpy as np
from src.data.episode_loader import EpisodeLoader, EpisodeSampler, manifest_overlap_report


class TestEpisodeLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'manifest.csv'
        self.rows = [
            dict(episode_id='train_005930_20230102', symbol='005930', split='train', start_date='2023-01-02', last_execution_date='2023-01-05', decision_steps=3),
            dict(episode_id='train_005930_20230103', symbol='005930', split='train', start_date='2023-01-03', last_execution_date='2023-01-06', decision_steps=3),
            dict(episode_id='test_005930_20230104', symbol='005930', split='test', start_date='2023-01-04', last_execution_date='2023-01-09', decision_steps=3),
        ]
        self.write_manifest()
        idx = pd.bdate_range('2023-01-02', periods=6)
        self.df = pd.DataFrame({'Open': 100., 'Close': 100., 'Tradable': True, 'FeatureReady': True,
            'return_1d': 0., 'return_5d': 0., 'return_20d': 0.,
            'volatility_20d': 0.1, 'ma_gap_20d': 0.}, index=idx)
        self.loader = EpisodeLoader(self.path, self.tmp.name)
        self.loader._frames['005930'] = self.df

    def write_manifest(self):
        pd.DataFrame(self.rows).to_csv(self.path, index=False)

    def test_symbol_preserves_leading_zeros(self):
        self.assertEqual(self.loader.record(self.rows[0]['episode_id']).symbol, '005930')

    def test_exact_window(self):
        frame = self.loader.load(self.rows[0]['episode_id'], expected_split='train')
        self.assertEqual(len(frame), 4)
        self.assertEqual(frame.index[-1], pd.Timestamp('2023-01-05'))

    def test_split_guard(self):
        with self.assertRaises(ValueError):
            self.loader.load(self.rows[2]['episode_id'], expected_split='train')

    def test_sampler_reproducible(self):
        a = EpisodeSampler(self.loader, seed=7).sample_ids(20)
        b = EpisodeSampler(self.loader, seed=7).sample_ids(20)
        self.assertEqual(a, b)
        self.assertTrue(all(x.startswith('train_') for x in a))

    def test_no_future_or_frame_mutation(self):
        w = self.loader.load(self.rows[0]['episode_id'])
        w.loc[w.index[0], 'Close'] = 999.
        self.assertEqual(self.df.iloc[0]['Close'], 100.)

    def test_bad_manifest_duplicate(self):
        self.rows.append(dict(self.rows[0]))
        self.write_manifest()
        with self.assertRaises(ValueError):
            EpisodeLoader(self.path, self.tmp.name)

    def test_end_mismatch_rejected(self):
        self.loader._records[self.rows[0]['episode_id']] = replace(self.loader.record(self.rows[0]['episode_id']), last_execution_date=pd.Timestamp('2023-01-09'))
        with self.assertRaises(ValueError):
            self.loader.load(self.rows[0]['episode_id'])

    def test_overlap_report(self):
        report = manifest_overlap_report(self.loader)
        self.assertEqual(report['train']['overlapping_episodes'], 1)
        self.assertEqual(report['test']['episodes'], 1)

    def test_missing_parquet_rejected(self):
        self.loader._frames.clear()
        with self.assertRaises(FileNotFoundError):
            self.loader.load(self.rows[0]['episode_id'])

    def test_unknown_split(self):
        with self.assertRaises(ValueError):
            EpisodeSampler(self.loader, 'other')


if __name__ == '__main__':
    unittest.main()
