"""Audit Kiwoom daily data across tickers without altering raw files.

Run: python -m src.data.validate_kiwoom_universe
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

SYMBOLS = ('005930','000660','035420','035720','005380','000270','055550','105560')
REQUIRED = {'Open','High','Low','Close','Volume','Tradable','NonTradingCandidate','InvalidOHLC'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbols', nargs='+', default=list(SYMBOLS))
    p.add_argument('--input-dir', default='data/raw/kiwoom')
    p.add_argument('--out-dir', default='data/reports/kiwoom')
    args = p.parse_args()
    inp, out = Path(args.input_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    loaded, rows, exceptions = {}, [], []
    for symbol in args.symbols:
        path = inp / f'{symbol}_validated.parquet'
        if not path.exists():
            rows.append({'symbol':symbol,'status':'MISSING_FILE'})
            continue
        try:
            df = pd.read_parquet(path)
            missing = REQUIRED - set(df.columns)
            if missing:
                raise ValueError(f'missing columns: {sorted(missing)}')
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError('Date index must be DatetimeIndex')
            df = df.sort_index()
            loaded[symbol] = df
            prices = df[['Open','High','Low','Close','Volume']]
            invalid = df['InvalidOHLC'].fillna(True).astype(bool)
            nontrading = df['NonTradingCandidate'].fillna(True).astype(bool)
            tradable = df['Tradable'].fillna(False).astype(bool)
            inconsistent = (tradable & (nontrading | invalid)).sum()
            summary = {
                'symbol':symbol,'status':'CHECK' if df.index.has_duplicates or inconsistent or prices.isna().any().any() else 'OK',
                'rows':len(df),'first':df.index.min().date().isoformat(),
                'last':df.index.max().date().isoformat(),
                'duplicate_dates':int(df.index.duplicated().sum()),
                'missing_ohlcv_cells':int(prices.isna().sum().sum()),
                'nontrading':int(nontrading.sum()),'invalid_ohlc':int(invalid.sum()),
                'tradable':int(tradable.sum()),'flag_conflicts':int(inconsistent),
                'adjustment_events':int(df['AdjustmentEvent'].fillna('').astype(str).str.strip().ne('').sum()) if 'AdjustmentEvent' in df else None,
            }
            rows.append(summary)
            bad = df.loc[nontrading | invalid, ['Open','High','Low','Close','Volume','NonTradingCandidate','InvalidOHLC','Tradable']].copy()
            if not bad.empty:
                bad.insert(0,'symbol',symbol)
                exceptions.append(bad.reset_index().rename(columns={df.index.name or 'index':'Date'}))
        except Exception as exc:
            rows.append({'symbol':symbol,'status':'ERROR','message':f'{type(exc).__name__}: {exc}'})

    # Union of dates is an INTERNAL cross-symbol reference, not an exchange calendar.
    if loaded:
        reference = pd.DatetimeIndex(sorted(set().union(*(set(x.index) for x in loaded.values()))))
        for row in rows:
            symbol = row['symbol']
            if symbol not in loaded:
                continue
            df = loaded[symbol]
            active_dates = reference[(reference >= df.index.min()) & (reference <= df.index.max())]
            missing_dates = active_dates.difference(df.index)
            row['missing_vs_union'] = len(missing_dates)
            for day in missing_dates:
                exceptions.append(pd.DataFrame([{'Date':day,'symbol':symbol,'Issue':'Missing_vs_union'}]))

    summary = pd.DataFrame(rows)
    summary.to_csv(out/'universe_quality_summary.csv',index=False,encoding='utf-8-sig')
    if exceptions:
        pd.concat(exceptions,ignore_index=True,sort=False).to_csv(out/'universe_exceptions.csv',index=False,encoding='utf-8-sig')
    else:
        pd.DataFrame(columns=['Date','symbol','Issue']).to_csv(out/'universe_exceptions.csv',index=False,encoding='utf-8-sig')
    print(summary.to_string(index=False))
    print(f'\nReports: {out.resolve()}')
    print('NOTE: missing_vs_union is NOT a certified KRX trading-calendar check.')
    if (summary['status'] != 'OK').any() or ('missing_vs_union' in summary and summary['missing_vs_union'].fillna(0).gt(0).any()):
        raise SystemExit('Review warnings in quality report')

if __name__ == '__main__':
    main()
