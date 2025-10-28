# data_loader.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Download universe-level OHLCV data
def download_data(tickers=None, years=15):
    if tickers is None:
        # Default — first 100 S&P tickers (saved by pair_selection)
        tickers = pd.read_csv("data/sp100.csv")["Symbol"].tolist()

    end = datetime.now()
    start = end - timedelta(days=years * 365)

    print(f"📡 Fetching {len(tickers)} tickers from Yahoo Finance...")

    df = yf.download(
        tickers,
        start=start,
        end=end,
        progress=True,
        group_by="ticker"
    )

    # ✅ Convert to simple wide DataFrame with Close prices
    closes = pd.DataFrame({
        t: df[t]["Close"] for t in tickers if t in df and "Close" in df[t]
    })

    closes.dropna(axis=1, how="all", inplace=True)
    closes.to_csv(f"{DATA_DIR}/all_closes.csv")

    print(f"✅ Downloaded {closes.shape[1]} tickers with {closes.shape[0]} rows of history")
    return closes

# ✅ Load only selected trading pair
def load_pair_prices(filepath="data/stocks.csv"):
    return pd.read_csv(filepath, index_col=0, parse_dates=True)

# ✅ Split dataset for walk-forward evaluation
def split_data(closes):
    n = len(closes)
    train_end = int(n * 0.6)
    test_end = int(n * 0.8)

    train = closes.iloc[:train_end]
    test = closes.iloc[train_end:test_end]
    valid = closes.iloc[test_end:]

    return train, test, valid
