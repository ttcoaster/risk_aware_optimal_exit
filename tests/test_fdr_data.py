import FinanceDataReader as fdr
import pandas as pd

TICKER = "005930"
START = "2005-01-01"
END = "2026-11-12"

df = fdr.DataReader(TICKER, START, END)

print("\n========== BASIC INFO ==========")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Index:", type(df.index))
print("Dtypes:\n", df.dtypes)

print("\n========== HEAD ==========")
print(df.head())

print("\n========== TAIL ==========")
print(df.tail())

print("\n========== MISSING VALUES ==========")
print(df.isna().sum())

print("\n========== DUPLICATED DATES ==========")
print(df.index.duplicated().sum())

print("\n========== DATE RANGE ==========")
print(df.index.min(), "~", df.index.max())

print("\n========== STOCK SPLIT PERIOD ==========")
print(df.loc["2018-04-25":"2018-05-10"])

print("\n========== PRICE VALIDATION ==========")

required = ["Open", "High", "Low", "Close", "Volume"]

print("Required columns:", all(c in df.columns for c in required))

for col in ["Open", "High", "Low", "Close"]:
    print(f"{col} <= 0:", (df[col] <= 0).sum())

print("High < Low:", (df["High"] < df["Low"]).sum())

print("Close outside High-Low:",
      ((df["Close"] > df["High"]) |
       (df["Close"] < df["Low"])).sum())

print("\n========== DONE ==========")