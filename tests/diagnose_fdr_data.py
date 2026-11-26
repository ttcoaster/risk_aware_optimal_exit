import FinanceDataReader as fdr
import pandas as pd

TICKER = "005930"

# 1. 기간별 데이터 요청
periods = [
    ("2005-01-01", "2010-01-01"),
    ("2010-01-01", "2015-01-01"),
    ("2015-01-01", "2020-01-01"),
    ("2020-01-01", "2026-11-13"),
]

frames = []

print("\n========== PERIOD TEST ==========")

for start, end in periods:
    try:
        part = fdr.DataReader(TICKER, start, end)

        print(
            f"{start} ~ {end}: "
            f"{len(part)} rows, "
            f"first={part.index.min()}, "
            f"last={part.index.max()}"
        )

        if not part.empty:
            frames.append(part)

    except Exception as e:
        print(f"ERROR {start} ~ {end}: {e}")

if not frames:
    raise RuntimeError("No data collected")

# 2. 데이터 병합
df = pd.concat(frames).sort_index()

duplicate_count = df.index.duplicated().sum()
df = df[~df.index.duplicated(keep="last")].copy()

print("\n========== MERGED DATA ==========")
print("Rows:", len(df))
print("Duplicate dates before removal:", duplicate_count)
print("First:", df.index.min())
print("Last:", df.index.max())

# 3. OHLC 이상치
price_cols = ["Open", "High", "Low", "Close"]

zero_prices = (df[price_cols] <= 0).any(axis=1)

invalid_ohlc = (
    (df["High"] < df["Low"])
    | (df["Open"] > df["High"])
    | (df["Open"] < df["Low"])
    | (df["Close"] > df["High"])
    | (df["Close"] < df["Low"])
)

print("\n========== ZERO PRICE ROWS ==========")
print(df.loc[zero_prices])

print("\n========== INVALID OHLC ROWS ==========")
print(df.loc[invalid_ohlc])

# 4. 거래정지 의심일
suspension_candidates = (
    (df["Volume"] == 0)
    & (df[["Open", "High", "Low"]] == 0).all(axis=1)
)

print("\n========== SUSPENSION CANDIDATES ==========")
print(df.loc[suspension_candidates])

# 5. 거래 가능한 날짜 후보
tradable = (
    (df["Open"] > 0)
    & (df["High"] > 0)
    & (df["Low"] > 0)
    & (df["Close"] > 0)
    & (df["Volume"] > 0)
    & ~invalid_ohlc
)

print("\n========== TRADABLE SUMMARY ==========")
print("Total rows:", len(df))
print("Tradable candidates:", tradable.sum())
print("Excluded candidates:", (~tradable).sum())

print("\n========== DONE ==========")