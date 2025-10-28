import os
import yfinance as yf
import pandas as pd
from pair_selection import get_us_tech50

def download_data(years: int = 15) -> pd.DataFrame:
    """
    Downloads 15 years of daily price data for the selected universe.
    """
    tickers = get_us_tech50()
    print(f"📥 Downloading market data for {len(tickers)} tickers...")

    data = yf.download(
        tickers,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        progress=True
    )["Close"]

    # Drop tickers that failed to download
    data = data.dropna(axis=1, how="all")
    print(f"✅ Data loaded. Valid tickers: {len(data.columns)}")

    os.makedirs("data", exist_ok=True)
    data.to_csv("data/all_prices.csv")
    print("💾 Saved → data/all_prices.csv")

    return data

def split_data(closes: pd.DataFrame):
    """
    Chronological 60/20/20 split.
    """
    n = len(closes)
    train = closes.iloc[: int(n * 0.60)].copy()
    test  = closes.iloc[int(n * 0.60): int(n * 0.80)].copy()
    valid = closes.iloc[int(n * 0.80):].copy()
    return train, test, valid

