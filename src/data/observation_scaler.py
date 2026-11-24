"""Train-only S1 feature scaling. No normalization of account state."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

FEATURES = ('return_1d','return_5d','return_20d','volatility_20d','ma_gap_20d')

@dataclass(frozen=True)
class FeatureScaler:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    clip: float = 10.0

    def __post_init__(self):
        if len(self.mean) != 5 or len(self.scale) != 5:
            raise ValueError('five feature statistics required')
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or any(v <= 0 for v in self.scale):
            raise ValueError('nonfinite or nonpositive statistics')
        if not np.isfinite(self.clip) or self.clip <= 0:
            raise ValueError('clip must be finite positive')

    @classmethod
    def fit(cls, values, *, clip=10.0):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 5 or len(arr) < 2 or not np.isfinite(arr).all():
            raise ValueError('expected finite matrix with >=2 rows and five columns')
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        return cls(tuple(map(float,mu)),tuple(map(float,sd)),float(clip))

    def transform(self, observation):
        arr = np.asarray(observation, dtype=np.float64)
        if arr.shape != (8,) or not np.isfinite(arr).all():
            raise ValueError('expected finite 8-dimensional observation')
        result = arr.copy()
        result[:5] = np.clip((result[:5] - self.mean) / self.scale, -self.clip, self.clip)
        if not np.isfinite(result).all():
            raise ValueError('nonfinite scaled observation')
        return result.astype(np.float32)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'version':1,'features':list(FEATURES),'mean':self.mean,'scale':self.scale,'clip':self.clip},indent=2),encoding='utf-8')

    @classmethod
    def load(cls,path: str | Path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload.get('version') != 1 or tuple(payload.get('features',())) != FEATURES:
            raise ValueError('scaler feature schema/version mismatch')
        return cls(tuple(payload['mean']),tuple(payload['scale']),float(payload['clip']))


def fit_train_scaler(loader, *, clip=10.0):
    """Fit on UNIQUE (symbol, date) observations inside train episode windows only.

    Uses only decision rows, not the extra terminal execution row. Excludes
    suspended/non-ready rows. No validation/test access.
    """
    by_symbol: dict[str,set] = {}
    for rec in loader.records('train'):
        window = loader.load(rec.episode_id, expected_split='train')
        by_symbol.setdefault(rec.symbol,set()).update(window.index[:-1])
    chunks=[]
    counts={}
    for symbol, dates in sorted(by_symbol.items()):
        frame = loader._frame(symbol)
        idx = frame.index.intersection(sorted(dates))
        rows = frame.loc[idx]
        valid = rows['Tradable'].astype(bool) & rows['FeatureReady'].astype(bool)
        data = rows.loc[valid,list(FEATURES)].to_numpy(dtype=np.float64)
        if len(data) and not np.isfinite(data).all():
            raise ValueError(f'nonfinite training features for {symbol}')
        counts[symbol] = len(data)
        if len(data): chunks.append(data)
    if not chunks: raise ValueError('no train observations')
    matrix=np.concatenate(chunks,axis=0)
    return FeatureScaler.fit(matrix,clip=clip),counts
