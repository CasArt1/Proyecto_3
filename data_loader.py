# ==========================================================
# data_loader.py
# ==========================================================
# Handles dynamic downloading and caching of Adjusted Close
# prices for selected tickers using Yahoo Finance.
# ==========================================================

import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_data(tickers, years=15, cache_path="data/stocks.csv"):
    """
    Downloads Adjusted Close prices for the given tickers from Yahoo Finance.
    If a cached file exists, it loads it instead of re-downloading.
    """
    os.makedirs("data", exist_ok=True)
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    # If cached file exists, reuse it
    if os.path.exists(cache_path):
        print(f"📂 Using cached data from {cache_path}")
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        return df

    print("⬇️ Downloading fresh data from Yahoo Finance...")
    data = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]

    # Clean data
    data = data.ffill().dropna(how="all")
    data.to_csv(cache_path)
    print(f"💾 Saved data to {cache_path}")

    return data

def load_data(cache_path="data/stocks.csv"):
    """
    Loads previously saved Adjusted Close data.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError("No cached data found. Run download_data() first.")
    return pd.read_csv(cache_path, index_col="Date", parse_dates=True)
