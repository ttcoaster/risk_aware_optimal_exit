"""STEP 4-1: reproducible, split-isolated episode loading (no source mutation)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

SPLITS = ('train', 'validation', 'test')
REQUIRED_MANIFEST = {'episode_id', 'symbol', 'split', 'start_date', 'last_execution_date', 'decision_steps'}
REQUIRED_FRAME = {'Open', 'Close', 'Tradable', 'FeatureReady'}
FEATURES = ('return_1d', 'return_5d', 'return_20d', 'volatility_20d', 'ma_gap_20d')


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    symbol: str
    split: str
    start_date: pd.Timestamp
    last_execution_date: pd.Timestamp
    decision_steps: int


class EpisodeLoader:
    """Loads exactly horizon+1 rows per manifest entry; caches parquet by symbol."""

    def __init__(self, manifest_path: str | Path, feature_dir: str | Path, *, cache: bool = True):
        self.manifest_path = Path(manifest_path)
        self.feature_dir = Path(feature_dir)
        manifest = pd.read_csv(self.manifest_path, dtype={'symbol': str, 'episode_id': str, 'split': str})
        missing = REQUIRED_MANIFEST - set(manifest.columns)
        if missing:
            raise ValueError(f'manifest missing columns: {sorted(missing)}')
        if manifest.empty or manifest['episode_id'].isna().any() or manifest['episode_id'].duplicated().any():
            raise ValueError('manifest must contain unique nonempty episode_id values')
        if not manifest['split'].isin(SPLITS).all():
            raise ValueError('unknown split in manifest')
        if manifest['symbol'].isna().any() or not manifest['symbol'].str.fullmatch(r'\d{6}').all():
            raise ValueError('symbols must be six-digit strings')
        for col in ('start_date', 'last_execution_date'):
            manifest[col] = pd.to_datetime(manifest[col], errors='raise')
            if manifest[col].isna().any():
                raise ValueError(f'missing {col}')
        steps = pd.to_numeric(manifest['decision_steps'], errors='raise')
        if steps.isna().any() or (steps < 1).any() or (steps % 1 != 0).any():
            raise ValueError('invalid decision_steps')
        manifest['decision_steps'] = steps.astype(int)
        if (manifest['start_date'] >= manifest['last_execution_date']).any():
            raise ValueError('invalid episode date range')
        self.manifest = manifest.copy()
        self._cache_enabled = bool(cache)
        self._frames: dict[str, pd.DataFrame] = {}
        self._records = {
            str(r.episode_id): EpisodeRecord(str(r.episode_id), str(r.symbol), str(r.split),
                pd.Timestamp(r.start_date), pd.Timestamp(r.last_execution_date), int(r.decision_steps))
            for r in manifest.itertuples(index=False)
        }

    def records(self, split: str) -> tuple[EpisodeRecord, ...]:
        if split not in SPLITS:
            raise ValueError(f'unknown split: {split}')
        return tuple(r for r in self._records.values() if r.split == split)

    def record(self, episode_id: str) -> EpisodeRecord:
        return self._records[episode_id]

    def _frame(self, symbol: str) -> pd.DataFrame:
        if symbol in self._frames:
            return self._frames[symbol]
        path = self.feature_dir / f'{symbol}_s1.parquet'
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='raise')
        if df.index.has_duplicates or not df.index.is_monotonic_increasing or df.index.hasnans:
            raise ValueError(f'bad date index for {symbol}')
        missing = (REQUIRED_FRAME | set(FEATURES)) - set(df.columns)
        if missing:
            raise ValueError(f'{symbol} missing columns: {sorted(missing)}')
        if self._cache_enabled:
            self._frames[symbol] = df
        return df

    def load(self, episode_id: str, *, expected_split: str | None = None) -> pd.DataFrame:
        r = self.record(episode_id)
        if expected_split is not None and r.split != expected_split:
            raise ValueError(f'split violation: requested {expected_split}, got {r.split}')
        df = self._frame(r.symbol)
        try:
            start = df.index.get_loc(r.start_date)
        except KeyError as exc:
            raise ValueError(f'episode start missing: {r.episode_id}') from exc
        if not isinstance(start, (int, np.integer)):
            raise ValueError('ambiguous start index')
        window = df.iloc[start:start + r.decision_steps + 1].copy()
        if len(window) != r.decision_steps + 1 or window.index[-1] != r.last_execution_date:
            raise ValueError(f'episode window mismatch: {r.episode_id}')
        if not bool(window.iloc[0]['FeatureReady']) or not bool(window.iloc[0]['Tradable']):
            raise ValueError(f'episode starts on non-ready/nontradable row: {r.episode_id}')
        return window

    def make_env(self, episode_id: str, *, expected_split: str | None = None,
                 initial_shares: int = 100, costs=None, reward_config=None):
        from src.envs.optimal_exit_env import OptimalExitEnv
        r = self.record(episode_id)
        return OptimalExitEnv(self.load(episode_id, expected_split=expected_split),
                              horizon=r.decision_steps, initial_shares=initial_shares,
                              costs=costs, reward_config=reward_config)


class EpisodeSampler:
    """Uniform episode sampling, reproducible with seed; never samples outside split."""

    def __init__(self, loader: EpisodeLoader, split: str = 'train', seed: int = 42):
        self.loader = loader
        self.split = split
        self.ids = tuple(r.episode_id for r in loader.records(split))
        if not self.ids:
            raise ValueError(f'no episodes for {split}')
        self.rng = np.random.default_rng(seed)

    def sample_id(self) -> str:
        return self.ids[int(self.rng.integers(len(self.ids)))]

    def sample(self) -> tuple[str, pd.DataFrame]:
        eid = self.sample_id()
        return eid, self.loader.load(eid, expected_split=self.split)

    def sample_ids(self, count: int) -> list[str]:
        if count < 0:
            raise ValueError('count must be nonnegative')
        return [self.sample_id() for _ in range(count)]


def manifest_overlap_report(loader: EpisodeLoader) -> dict:
    """Date ranges per split and within-split overlap counts (metadata only)."""
    result = {}
    for split in SPLITS:
        records = loader.records(split)
        by_symbol: dict[str, list[EpisodeRecord]] = {}
        for r in records:
            by_symbol.setdefault(r.symbol, []).append(r)
        overlaps = 0
        for group in by_symbol.values():
            group.sort(key=lambda x: x.start_date)
            max_end = pd.Timestamp.min
            for r in group:
                if r.start_date <= max_end:
                    overlaps += 1
                max_end = max(max_end, r.last_execution_date)
        result[split] = {'episodes': len(records), 'symbols': len(by_symbol),
                         'overlapping_episodes': overlaps,
                         'first_start': str(min((r.start_date for r in records), default=pd.NaT).date()) if records else None,
                         'last_execution': str(max((r.last_execution_date for r in records), default=pd.NaT).date()) if records else None}
    return result
