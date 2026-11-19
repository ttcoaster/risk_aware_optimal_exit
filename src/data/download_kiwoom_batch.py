
"""Batch downloader for Kiwoom daily OHLCV.

Run from project root:
python -m src.data.download_kiwoom_batch --symbols 005930 000660
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.kiwoom_daily import (
    REAL_HOST,
    MOCK_HOST,
    KiwoomAPIError,
    issue_token,
    fetch_daily,
    audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_credentials(mode: str):
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=env_path)

    prefix = "KIWOOM_REAL" if mode == "real" else "KIWOOM_DEMO"
    appkey = os.getenv(f"{prefix}_APP_KEY")
    secretkey = os.getenv(f"{prefix}_APP_SECRET")

    if not appkey or not secretkey:
        raise RuntimeError(
            f"Missing {prefix}_APP_KEY or {prefix}_APP_SECRET. "
            f"Check {env_path}"
        )

    return appkey, secretkey


def validate_symbols(symbols):
    unique = []

    for symbol in symbols:
        symbol = str(symbol).strip()

        if len(symbol) != 6 or not symbol.isdigit():
            raise ValueError(
                f"Invalid symbol: {symbol!r}. "
                "Expected six digits."
            )

        if symbol not in unique:
            unique.append(symbol)

    return unique


def existing_covers_range(path, start, end):
    """Conservative skip check using saved index bounds.

    This does not prove exchange-calendar completeness.
    """
    if not path.exists():
        return False

    try:
        df = pd.read_parquet(path)

        if df.empty or df.index.has_duplicates:
            return False

        first = pd.Timestamp(df.index.min())
        last = pd.Timestamp(df.index.max())

        # Allow the first available date to be shortly
        # after a non-trading requested start date.
        start_tolerance = pd.Timestamp(start) + pd.Timedelta(days=4)

        # Exact requested end date must be present or exceeded.
        return (
            first <= start_tolerance
            and last >= pd.Timestamp(end)
            and {"Open", "High", "Low", "Close", "Volume"}
                .issubset(df.columns)
        )
    except (OSError, ValueError, KeyError, ImportError):
        return False


def save_dataset(df, symbol, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"{symbol}_adjusted.parquet"
    validated_path = out_dir / f"{symbol}_validated.parquet"
    audit_path = out_dir / f"{symbol}_audit.csv"

    diagnostic = audit(df)

    audit_mask = (
        diagnostic["NonTradingCandidate"]
        | diagnostic["InvalidOHLC"]
    )

    # Temporary files avoid replacing a good file with
    # an incompletely written file.
    raw_tmp = raw_path.with_suffix(".parquet.tmp")
    validated_tmp = validated_path.with_suffix(".parquet.tmp")
    audit_tmp = audit_path.with_suffix(".csv.tmp")

    df.to_parquet(raw_tmp)
    diagnostic.to_parquet(validated_tmp)
    diagnostic.loc[audit_mask].to_csv(
        audit_tmp, encoding="utf-8-sig"
    )

    os.replace(raw_tmp, raw_path)
    os.replace(validated_tmp, validated_path)
    os.replace(audit_tmp, audit_path)

    return {
        "rows": len(df),
        "first_date": str(df.index.min().date()),
        "last_date": str(df.index.max().date()),
        "non_trading": int(
            diagnostic["NonTradingCandidate"].sum()
        ),
        "invalid_ohlc": int(
            diagnostic["InvalidOHLC"].sum()
        ),
        "tradable": int(diagnostic["Tradable"].sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch download Kiwoom adjusted daily OHLCV"
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Six-digit stock codes, separated by spaces",
    )
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument(
        "--mode",
        choices=["real", "demo"],
        default="real",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw/kiwoom",
    )
    parser.add_argument("--delay", type=float, default=1.1)
    parser.add_argument("--max-pages", type=int, default=300)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if existing date bounds match",
    )

    args = parser.parse_args()

    symbols = validate_symbols(args.symbols)

    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)

    if start_dt > end_dt or end_dt.date() > date.today():
        parser.error("Require start <= end <= today")

    if args.delay < 1.0:
        parser.error("--delay must be >= 1.0 seconds")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    appkey, secretkey = load_credentials(args.mode)
    host = REAL_HOST if args.mode == "real" else MOCK_HOST

    results = []

    with requests.Session() as session:
        token = issue_token(
            session, host, appkey, secretkey
        )

        print("OAuth authenticated (token hidden)")
        print(f"Symbols: {symbols}")

        for index, symbol in enumerate(symbols, start=1):
            print(
                f"\n[{index}/{len(symbols)}] "
                f"Processing {symbol}"
            )

            raw_path = out_dir / f"{symbol}_adjusted.parquet"
            validated_path = (
                out_dir / f"{symbol}_validated.parquet"
            )

            if (
                not args.force
                and existing_covers_range(
                    raw_path, args.start, args.end
                )
                and validated_path.exists()
            ):
                print("SKIPPED: existing data covers date bounds")
                results.append({
                    "symbol": symbol,
                    "status": "SKIPPED",
                    "rows": "",
                    "first_date": "",
                    "last_date": "",
                    "non_trading": "",
                    "invalid_ohlc": "",
                    "tradable": "",
                    "message": "Existing date bounds covered",
                })
                continue

            try:
                # Conservative pause between symbols.
                if index > 1:
                    time.sleep(args.delay)

                df = fetch_daily(
                    session=session,
                    host=host,
                    token=token,
                    symbol=symbol,
                    start=args.start,
                    end=args.end,
                    delay=args.delay,
                    max_pages=args.max_pages,
                    on_page=lambda p, n, oldest: print(
                        f"  page={p}, rows={n}, oldest={oldest}"
                    ),
                )

                if df.empty:
                    raise KiwoomAPIError("No data returned")

                stats = save_dataset(
                    df, symbol, out_dir
                )

                print(
                    f"SAVED: rows={stats['rows']}, "
                    f"first={stats['first_date']}, "
                    f"last={stats['last_date']}, "
                    f"non_trading={stats['non_trading']}, "
                    f"invalid_ohlc={stats['invalid_ohlc']}"
                )

                results.append({
                    "symbol": symbol,
                    "status": "SUCCESS",
                    **stats,
                    "message": "",
                })

            except (
                KiwoomAPIError,
                requests.RequestException,
                OSError,
                ValueError,
            ) as exc:
                print(
                    f"FAILED: {type(exc).__name__}: {exc}"
                )

                results.append({
                    "symbol": symbol,
                    "status": "FAILED",
                    "rows": "",
                    "first_date": "",
                    "last_date": "",
                    "non_trading": "",
                    "invalid_ohlc": "",
                    "tradable": "",
                    "message": str(exc),
                })

    summary = pd.DataFrame(results)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"batch_summary_{stamp}.csv"

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== BATCH SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nSummary saved: {summary_path}")

    if (summary["status"] == "FAILED").any():
        raise SystemExit(
            "Some symbols failed. Review the summary CSV."
        )


if __name__ == "__main__":
    main()
