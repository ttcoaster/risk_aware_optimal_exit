"""Create a deterministic manifest of 60-decision optimal-exit episodes.

Each episode references 60 decision rows and the following 61st row for
next-open execution. No price paths, actions or rewards are precomputed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

SYMBOLS = ('005930','000660','035420','035720','005380','000270','055550','105560')
SPLITS = {
    'train': ('2005-01-01', '2018-12-31'),
    'validation': ('2019-01-01', '2022-12-31'),
    'test': ('2023-01-01', '2026-11-12'),
}
REQUIRED = {'Open','High','Low','Close','Volume','Tradable','FeatureReady',
            'return_1d','return_5d','return_20d','volatility_20d','ma_gap_20d'}


def validate_frame(df: pd.DataFrame, symbol: str) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f'{symbol}: expected DatetimeIndex')
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError(f'{symbol}: dates must be unique and sorted')
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f'{symbol}: missing columns {sorted(missing)}')
    if df.empty:
        raise ValueError(f'{symbol}: empty input')
    if df[list(REQUIRED - {'Tradable','FeatureReady'})].isna().all(axis=0).any():
        raise ValueError(f'{symbol}: entirely missing required numeric column')
    for flag in ('Tradable','FeatureReady'):
        if df[flag].isna().any() or not df[flag].isin([True,False]).all():
            raise ValueError(f'{symbol}: invalid {flag} flag')
    if (df['FeatureReady'] & ~df['Tradable']).any():
        raise ValueError(f'{symbol}: FeatureReady=True on nontradable row')
    feature_cols = ['return_1d','return_5d','return_20d','volatility_20d','ma_gap_20d']
    if df.loc[df['FeatureReady'],feature_cols].isna().any().any():
        raise ValueError(f'{symbol}: NaN feature on FeatureReady row')


def build_manifest_for_symbol(df: pd.DataFrame, symbol: str, *, horizon=60, stride=20,
                              splits=None) -> list[dict]:
    if horizon < 1 or stride < 1:
        raise ValueError('horizon and stride must be positive')
    validate_frame(df, symbol)
    splits = SPLITS if splits is None else splits
    records = []
    for split, (start, end) in splits.items():
        period = df.loc[pd.Timestamp(start):pd.Timestamp(end)]
        # horizon decision rows + one next-open execution/valuation row
        if len(period) < horizon + 1:
            continue
        for offset in range(0, len(period) - horizon, stride):
            decision_rows = period.iloc[offset:offset+horizon]
            next_open_row = period.iloc[offset+horizon]
            # Only the entry date must be feature-ready and tradable.
            # Later suspensions stay in the episode, for the environment to handle.
            if not bool(decision_rows['FeatureReady'].iloc[0]):
                continue
            if not bool(decision_rows['Tradable'].iloc[0]):
                continue
            records.append({
                'episode_id': f'{split}_{symbol}_{decision_rows.index[0]:%Y%m%d}',
                'symbol': symbol,
                'split': split,
                'start_date': decision_rows.index[0].strftime('%Y-%m-%d'),
                'last_decision_date': decision_rows.index[-1].strftime('%Y-%m-%d'),
                'last_execution_date': next_open_row.name.strftime('%Y-%m-%d'),
                'decision_steps': horizon,
                'nontrading_decision_rows': int((~decision_rows['Tradable'].astype(bool)).sum()),
                'last_execution_tradable': bool(next_open_row['Tradable']),
            })
    return records


def build_all(input_dir: Path, symbols, horizon=60, stride=20, splits=None):
    rows = []
    for symbol in symbols:
        file = input_dir / f'{symbol}_s1.parquet'
        if not file.is_file():
            raise FileNotFoundError(f'Missing input: {file}')
        df = pd.read_parquet(file)
        new = build_manifest_for_symbol(df, symbol, horizon=horizon, stride=stride, splits=splits)
        print(f'{symbol}: {len(new)} episodes')
        rows.extend(new)
    if not rows:
        raise ValueError('No episodes generated')
    manifest = pd.DataFrame(rows)
    if manifest['episode_id'].duplicated().any():
        raise ValueError('Duplicate episode IDs')
    return manifest.sort_values(['split','symbol','start_date']).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', default='data/processed/s1')
    p.add_argument('--out-dir', default='data/processed/episodes')
    p.add_argument('--symbols', nargs='+', default=list(SYMBOLS))
    p.add_argument('--horizon', type=int, default=60)
    p.add_argument('--stride', type=int, default=20)
    args = p.parse_args()
    if args.horizon < 1 or args.stride < 1:
        p.error('horizon and stride must be >= 1')
    symbols = list(dict.fromkeys(args.symbols))
    if any(len(s)!=6 or not s.isdigit() for s in symbols):
        p.error('symbols must be six-digit strings')
    manifest = build_all(Path(args.input_dir), symbols, args.horizon, args.stride)
    out = Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    manifest.to_csv(out/'episode_manifest.csv',index=False,encoding='utf-8-sig')
    manifest.to_parquet(out/'episode_manifest.parquet',index=False)
    summary = manifest.groupby(['split','symbol'],sort=False).agg(
        episodes=('episode_id','size'),
        first_start=('start_date','min'),
        last_start=('start_date','max'),
        episodes_with_nontrading=('nontrading_decision_rows',lambda x:int((x>0).sum())),
        final_execution_nontrading=('last_execution_tradable',lambda x:int((~x).sum()))
    ).reset_index()
    summary.to_csv(out/'episode_summary.csv',index=False,encoding='utf-8-sig')
    print('\n=== SUMMARY ===')
    print(summary.to_string(index=False))
    print(f'\nTotal episodes: {len(manifest)}')
    print(f'Saved to: {out.resolve()}')

if __name__ == '__main__':
    main()
