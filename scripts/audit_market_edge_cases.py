"""Read-only audit of STEP 2-7 S1 parquet files for STEP 3-5.
Run from project root: python -m scripts.audit_market_edge_cases
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [ROOT / 'data' / 'processed' / 's1', ROOT / 'data' / 'processed']


def main():
    paths = []
    for directory in SEARCH_DIRS:
        if directory.is_dir():
            paths = sorted(directory.rglob('*_s1.parquet'))
            if paths:
                break
    if not paths:
        raise FileNotFoundError('No *_s1.parquet found under data/processed/s1 or data/processed')
    print('STEP 3-5 READ-ONLY DATA AUDIT')
    print('Files:', len(paths))
    total_nontradable = 0
    total_inconsistent = 0
    for path in paths:
        df = pd.read_parquet(path)
        required = {'Open','Close','Volume','Tradable','FeatureReady'}
        missing = required.difference(df.columns)
        if missing:
            print(f'{path.name}: MISSING {sorted(missing)}')
            continue
        tradable = df['Tradable'].fillna(False).astype(bool)
        suspended = ~tradable
        bad_close = (~np.isfinite(pd.to_numeric(df['Close'], errors='coerce'))) | (df['Close'] <= 0)
        bad_open_tradable = tradable & ((~np.isfinite(pd.to_numeric(df['Open'], errors='coerce'))) | (df['Open'] <= 0))
        ready_suspended = suspended & df['FeatureReady'].fillna(False).astype(bool)
        zero_volume_tradable = tradable & (df['Volume'] <= 0)
        inconsistencies = int(bad_close.sum() + bad_open_tradable.sum() + ready_suspended.sum() + zero_volume_tradable.sum())
        total_nontradable += int(suspended.sum())
        total_inconsistent += inconsistencies
        print(f'{path.name}: rows={len(df)}, nontradable={int(suspended.sum())}, '
              f'bad_close={int(bad_close.sum())}, bad_open_tradable={int(bad_open_tradable.sum())}, '
              f'ready_suspended={int(ready_suspended.sum())}, zero_volume_tradable={int(zero_volume_tradable.sum())}')
        if suspended.any():
            print('  nontradable dates (first 10):', ', '.join(str(d.date()) for d in df.index[suspended][:10]))
    print('TOTAL nontradable rows:', total_nontradable)
    print('TOTAL basic inconsistencies:', total_inconsistent)
    print('NOTE: No orders are executed and no source files are modified.')
    print('NOTE: Adjusted OHLCV alone cannot establish cash dividends or share-count transformations.')


if __name__ == '__main__':
    main()
