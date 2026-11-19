"""Build five causal, price-based S1 features from Kiwoom validated daily data.

Usage (from repository root):
    python -m src.features.build_s1_features
    python -m src.features.build_s1_features --symbols 005930 000660

Only dates with Tradable=True contribute to the trading-session rolling windows.
Non-tradable source rows are retained with NaN feature values. No forward filling.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ("005930", "000660", "035420", "035720", "005380", "000270", "055550", "105560")
FEATURES = ("return_1d", "return_5d", "return_20d", "volatility_20d", "ma_gap_20d")
REQUIRED = {"Open", "High", "Low", "Close", "Volume", "Tradable"}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return source rows plus five features and a FeatureReady flag.

    20-day volatility = sample std (ddof=1) of 20 consecutive log returns.
    The first fully ready row needs 21 valid trading-session closes.
    """
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Date index must be sorted ascending and unique")
    if df.empty:
        raise ValueError("Input is empty")
    if df["Tradable"].isna().any() or not pd.api.types.is_bool_dtype(df["Tradable"]):
        raise ValueError("Tradable must be boolean with no missing values")
    if df.loc[df["Tradable"], "Close"].isna().any() or (df.loc[df["Tradable"], "Close"] <= 0).any():
        raise ValueError("Tradable close prices must be positive and nonmissing")

    result = df.copy()
    for feature in FEATURES:
        result[feature] = np.nan

    # Non-trading rows do not count toward the lookback window.
    # Do not bridge a genuine data gap silently: calendar completeness is
    # a separate data-validation responsibility.
    traded = result.loc[result["Tradable"]].copy()
    close = traded["Close"].astype("float64")
    log_returns = np.log(close).diff()
    traded["return_1d"] = close.pct_change(periods=1, fill_method=None)
    traded["return_5d"] = close.pct_change(periods=5, fill_method=None)
    traded["return_20d"] = close.pct_change(periods=20, fill_method=None)
    traded["volatility_20d"] = log_returns.rolling(window=20, min_periods=20).std(ddof=1)
    traded["ma_gap_20d"] = close / close.rolling(window=20, min_periods=20).mean() - 1.0
    result.loc[traded.index, list(FEATURES)] = traded[list(FEATURES)]
    result["FeatureReady"] = result["Tradable"] & result[list(FEATURES)].notna().all(axis=1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal S1 price features from Kiwoom data")
    parser.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    parser.add_argument("--input-dir", default="data/raw/kiwoom")
    parser.add_argument("--output-dir", default="data/processed/s1")
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    failed = False
    for symbol in args.symbols:
        if len(symbol) != 6 or not symbol.isdigit():
            parser.error(f"Invalid symbol: {symbol}")
        source = input_dir / f"{symbol}_validated.parquet"
        target = output_dir / f"{symbol}_s1.parquet"
        try:
            df = pd.read_parquet(source)
            result = build_features(df)
            result.to_parquet(target)
            ready = result.loc[result["FeatureReady"]]
            summaries.append({
                "symbol": symbol, "status": "OK", "rows": len(result),
                "tradable": int(result["Tradable"].sum()),
                "feature_ready": len(ready),
                "first_ready": str(ready.index.min().date()) if len(ready) else "",
                "nontrading": int((~result["Tradable"]).sum()),
                "message": "",
            })
            print(f"{symbol}: rows={len(result)}, ready={len(ready)}, saved={target}")
        except (OSError, ValueError, KeyError, ImportError) as exc:
            failed = True
            summaries.append({"symbol": symbol, "status": "FAILED", "message": str(exc)})
            print(f"{symbol}: FAILED: {exc}")

    report = output_dir / "s1_feature_summary.csv"
    pd.DataFrame(summaries).to_csv(report, index=False, encoding="utf-8-sig")
    print(f"Summary: {report}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
