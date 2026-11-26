# E1 Kiwoom REST daily OHLCV (read-only)

This standalone module uses **Python 3.11+**, `requests`, `pandas`, and `pyarrow`. No PyQt, COM, official 3.13 SDK, CLI, or trading/order endpoints.

## Installation in existing conda env

```powershell
conda activate e1-optimal-exit
python -m pip install requests pyarrow
python -m pip check
```

## Credentials: PowerShell session (real key)

```powershell
$env:KIWOOM_REAL_APP_KEY = Read-Host 'Kiwoom REAL App Key'
$env:KIWOOM_REAL_APP_SECRET = Read-Host 'Kiwoom REAL App Secret'
```

Use **real API access for historical chart retrieval** only; no order code exists. Never share these values or save in a tracked file. Do not confuse the Python env with Kiwoom real/demo account mode.

## Run from repository root

Copy `src/data/kiwoom_daily.py` into your repo's `src/data/`, and `tests/test_kiwoom_daily.py` into `tests/`. Create `src/__init__.py` and `src/data/__init__.py` if needed.

```powershell
python -m unittest discover -s tests -p test_kiwoom_daily.py -v
python -m src.data.kiwoom_daily --symbol 005930 --start 2005-01-01 --end 2026-11-12 --mode real
```

Results: `data/raw/kiwoom/005930_adjusted.parquet`, `005930_audit.csv`. The Parquet includes OHLCV and raw adjustment event fields. It is **not yet a certified total-return/fully corporate-action-adjusted backtest dataset**. Verify 2018 split, dividend handling, zero-price rows, original OHLC consistency, and early history before training.

## Notes

- The API PDF documents `au10001` (p.12) and `ka10081` (pp.220-222).
- The daily chart API requires a base date after a corporate action for adjusted prices; pagination preserves the same recent base date.
- Requests are sequential with a conservative >=1 second pause. No automated retries on rate-limit errors.
- Never interpret a missing 2005 history as proof that it exists; check the earliest returned date.
- Do not push credentials, access tokens, or personal account data to Git.
