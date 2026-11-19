"""Kiwoom REST API daily OHLCV downloader (read-only, Python 3.11+).

API: au10001 and ka10081, per uploaded Kiwoom REST API PDF pp.12, 220-222.
Never logs or persists credentials/tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import os
from dotenv import load_dotenv




REAL_HOST = "https://api.kiwoom.com"
MOCK_HOST = "https://mockapi.kiwoom.com"
API_ID = "ka10081"
DATA_KEY = "stk_dt_pole_chart_qry"
FIELDS = {"dt": "Date", "open_pric": "Open", "high_pric": "High", "low_pric": "Low", "cur_prc": "Close", "trde_qty": "Volume"}


class KiwoomAPIError(RuntimeError):
    pass


def _post(session, url, *, headers, body, timeout=25):
    try:
        response = session.post(url, headers=headers, json=body, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise KiwoomAPIError(f"HTTP/network error: {type(exc).__name__}: {exc}") from exc
    except ValueError as exc:
        raise KiwoomAPIError("Response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise KiwoomAPIError("Unexpected response body type")
    code = payload.get("return_code")
    if code is None or str(code) != "0":
        raise KiwoomAPIError(f"Kiwoom API error: code={code}, msg={payload.get('return_msg')}")
    return response, payload


def issue_token(session, host, appkey, secretkey):
    _, payload = _post(session, host + "/oauth2/token", headers={"Content-Type": "application/json;charset=UTF-8"}, body={"grant_type": "client_credentials", "appkey": appkey, "secretkey": secretkey})
    token = payload.get("token")
    if not token:
        raise KiwoomAPIError("Token missing from OAuth response")
    return token


def parse_daily(rows):
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "AdjustmentEvent", "AdjustmentRatio"]).rename_axis("Date")
    raw = pd.DataFrame(rows)
    absent = set(FIELDS) - set(raw.columns)
    if absent:
        raise KiwoomAPIError(f"Missing daily response fields: {sorted(absent)}")
    df = raw.rename(columns=FIELDS).copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="raise")
    for name in ("Open", "High", "Low", "Close", "Volume"):
        # Kiwoom chart prices may contain sign prefixes. The magnitude is the price.
        values = df[name].astype(str).str.replace(",", "", regex=False).str.strip()
        df[name] = pd.to_numeric(values, errors="raise").abs().astype("int64")
    df["AdjustmentEvent"] = raw.get("upd_stkpc_tp", pd.Series("", index=raw.index)).fillna("").astype(str).to_numpy()
    df["AdjustmentRatio"] = raw.get("upd_rt", pd.Series("", index=raw.index)).fillna("").astype(str).to_numpy()
    df = df.set_index("Date").sort_index()
    # Duplicates at pagination boundaries should have identical values; otherwise fail.
    dup = df.index.duplicated(keep=False)
    if dup.any():
        for dt, part in df.loc[dup].groupby(level=0):
            if (part.nunique(dropna=False) > 1).any():
                raise KiwoomAPIError(f"Conflicting duplicate daily records at {dt.date()}")
        df = df.loc[~df.index.duplicated(keep="first")]
    return df[["Open", "High", "Low", "Close", "Volume", "AdjustmentEvent", "AdjustmentRatio"]]


def fetch_daily(session, host, token, symbol, start, end, *, delay=1.1, max_pages=300, on_page=None):
    if not re.fullmatch(r"\d{6}", symbol):
        raise ValueError("symbol must be a six-digit Korean stock code")
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    if start_dt > end_dt or end_dt.date() > date.today():
        raise ValueError("Invalid date range: start <= end <= today")
    if delay < 1.0:
        raise ValueError("Use >=1.0 second between requests for conservative rate limiting")
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    records = []
    cont_yn, next_key = "N", ""
    seen_keys = set()
    completed = False
    oldest_seen = None
    for page in range(1, max_pages + 1):
        if page > 1:
            time.sleep(delay)
        headers = {"Content-Type": "application/json;charset=UTF-8", "authorization": f"Bearer {token}", "api-id": API_ID, "cont-yn": cont_yn, "next-key": next_key}
        response, payload = _post(session, host + "/api/dostk/chart", headers=headers, body={"stk_cd": symbol, "base_dt": end_dt.strftime("%Y%m%d"), "upd_stkpc_tp": "1"})
        chunk = payload.get(DATA_KEY)
        if not isinstance(chunk, list):
            raise KiwoomAPIError(f"Expected {DATA_KEY} list; got {type(chunk).__name__}")
        records.extend(chunk)
        dates = [r.get("dt", "") for r in chunk if isinstance(r, dict)]
        if dates:
            oldest_seen = min(dates) if oldest_seen is None else min(oldest_seen, min(dates))
        if on_page:
            on_page(page, len(chunk), oldest_seen)
        has_next = response.headers.get("cont-yn", "N").upper() == "Y"
        candidate_key = response.headers.get("next-key", "")
        if oldest_seen and oldest_seen <= start_dt.strftime("%Y%m%d"):
            completed = True
            break
        if not has_next:
            completed = True
            break
        if not candidate_key or candidate_key in seen_keys or not chunk:
            raise KiwoomAPIError("Invalid pagination: missing/repeated next-key or empty continuation page")
        seen_keys.add(candidate_key)
        cont_yn, next_key = "Y", candidate_key
    if not completed:
        raise KiwoomAPIError(f"max_pages={max_pages} reached; refusing to save potentially incomplete data")
    df = parse_daily(records)
    df = df.loc[start_dt:end_dt]
    return df



def audit(df):
    result = df.copy()

    # 거래정지 또는 매매 불가능일 후보
    # Volume=0이더라도 OHLC는 양수일 수 있음
    result["NonTradingCandidate"] = (
        (result["Volume"] == 0)
        | (result[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    )

    # 기존 컬럼명과의 호환성 유지
    result["SuspensionCandidate"] = result["NonTradingCandidate"]

    # OHLC 가격 정합성 검사
    result["InvalidOHLC"] = (
        (result[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        | (result["High"] < result["Low"])
        | (result["Open"] > result["High"])
        | (result["Open"] < result["Low"])
        | (result["Close"] > result["High"])
        | (result["Close"] < result["Low"])
    )

    # 매매 불가능 후보와 별개인 가격 오류
    result["PriceMismatch"] = (
        result["InvalidOHLC"]
        & ~result["NonTradingCandidate"]
    )

    # E1 환경에서 사용할 잠정적 매매 가능 여부
    result["Tradable"] = (
        (result["Volume"] > 0)
        & (result["Open"] > 0)
        & (result["High"] > 0)
        & (result["Low"] > 0)
        & (result["Close"] > 0)
        & ~result["InvalidOHLC"]
    )

    return result



def main():
    # 프로젝트 루트의 .env 파일 로드
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"

    load_dotenv(dotenv_path=env_path)
    parser = argparse.ArgumentParser(description="Download adjusted Kiwoom daily OHLCV (read-only)")
    parser.add_argument("--symbol", default="005930")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--mode", choices=["real", "demo"], default="real")
    parser.add_argument("--out-dir", default="data/raw/kiwoom")
    parser.add_argument("--delay", type=float, default=1.1)
    parser.add_argument("--max-pages", type=int, default=300)
    args = parser.parse_args()
    prefix = "KIWOOM_REAL" if args.mode == "real" else "KIWOOM_DEMO"
    appkey, secretkey = os.getenv(prefix + "_APP_KEY"), os.getenv(prefix + "_APP_SECRET")
    if not appkey or not secretkey:
        raise SystemExit(f"Set {prefix}_APP_KEY and {prefix}_APP_SECRET in your PowerShell session (never commit credentials)")
    host = REAL_HOST if args.mode == "real" else MOCK_HOST
    with requests.Session() as session:
        token = issue_token(session, host, appkey, secretkey)
        print("OAuth authenticated (token not displayed)")
        df = fetch_daily(session, host, token, args.symbol, args.start, args.end, delay=args.delay, max_pages=args.max_pages, on_page=lambda p, n, oldest: print(f"page={p} rows={n} oldest={oldest}"))
    if df.empty:
        raise SystemExit("No data returned; nothing saved")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    file = out / f"{args.symbol}_adjusted.parquet"

    # 원본 수정 OHLCV는 그대로 보존
    df.to_parquet(file)

    # 감사 정보 생성
    diagnostic = audit(df)

    # 이상치 또는 매매 불가능 후보만 CSV로 저장
    report = out / f"{args.symbol}_audit.csv"
    audit_mask = (
        diagnostic["NonTradingCandidate"]
        | diagnostic["InvalidOHLC"]
    )

    diagnostic.loc[audit_mask].to_csv(
        report,
        encoding="utf-8-sig"
    )

    # 매매 가능 여부를 포함한 별도 데이터 저장
    validated_file = out / f"{args.symbol}_validated.parquet"
    diagnostic.to_parquet(validated_file)

    print(f"Saved: {file} | rows={len(df)} | first={df.index.min().date()} | last={df.index.max().date()}")
    print(
    f"Non-trading candidates={diagnostic['NonTradingCandidate'].sum()}, "
    f"Invalid OHLC={diagnostic['InvalidOHLC'].sum()}, "
    f"Tradable={diagnostic['Tradable'].sum()}"
    )
    print(f"Validated: {validated_file}")
    print(f"Audit: {report}")
    print("Split/adjustment event sample:")
    print(df.loc["2018-04-25":"2018-05-10"].to_string())
    requested_start = pd.Timestamp(args.start)
    first_available = df.index.min()

    if first_available > requested_start:
        print(
            f"INFO: Requested start={requested_start.date()}, "
            f"first available date={first_available.date()}. "
            "Check listing date and trading calendar if needed."
        )

    
if __name__ == "__main__":
    main()
