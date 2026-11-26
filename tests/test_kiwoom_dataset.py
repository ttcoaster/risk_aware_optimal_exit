import pandas as pd

df = pd.read_parquet(
    "data/raw/kiwoom/005930_adjusted.parquet"
)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Date range:", df.index.min(), df.index.max())
print("Duplicate dates:", df.index.duplicated().sum())
print("Missing values:\n", df.isna().sum())

zero_volume = df[df["Volume"] == 0]

print("\n=== ZERO VOLUME ===")
print("Count:", len(zero_volume))
print(zero_volume.to_string())

print("\n=== ADJUSTMENT EVENTS ===")
events = df[
    df["AdjustmentEvent"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
]
print("Count:", len(events))
print(events.to_string())

print("\n=== OHLC VALIDATION ===")
invalid = (
    (df["High"] < df["Low"])
    | (df["Open"] > df["High"])
    | (df["Open"] < df["Low"])
    | (df["Close"] > df["High"])
    | (df["Close"] < df["Low"])
)

print("Invalid OHLC:", invalid.sum())
print(df.loc[invalid].to_string())
