# ==========================================================
# data_loader.py  ✅ Updated for latest yfinance changes
# ==========================================================
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


# ============================================
# Internal utility for robust CSV reading
# ============================================
def _read_csv_any(cache_path: str) -> pd.DataFrame:
    df0 = pd.read_csv(cache_path, nrows=5)
    cols = df0.columns.tolist()

    if "Date" in cols:
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=["Date"])
    else:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index.name = "Date"

    return df.sort_index().ffill().dropna(how="all")


# ============================================
# ✅ Download full universe in memory (no cache)
# ============================================
def download_data(tickers, years=15, cache_path=None):
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    # ✅ Only use cache if requested AND cache contains all tickers
    if cache_path and os.path.exists(cache_path):
        print(f"📂 Found cache at {cache_path} — validating tickers...")
        cached = _read_csv_any(cache_path)
        if all(t in cached.columns for t in tickers):
            print("✅ Cache contains full universe. Using it.")
            return cached[tickers]
        else:
            print("⚠️ Cache missing tickers. Downloading fresh...")

    print("⬇️ Downloading fresh data from Yahoo Finance...")
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,    # ✅ newest yfinance default behavior
        progress=False
    )

    # ✅ Some downloads return MultiIndex (e.g., Open/High/Low/Close)
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            raise RuntimeError("❌ Could not find Close prices in MultiIndex download.")
    else:
        data = data.rename(columns=lambda x: x.strip())

    data = data.ffill().dropna(how="all")

    print(f"📈 Downloaded {data.shape[1]} tickers with {data.shape[0]} rows.")
    return data


# ============================================
# ✅ Load only the saved optimal pair later on
# ============================================
def load_data(cache_path="data/stocks.csv"):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"No cached pair found at {cache_path}. Run main.py first.")
    print(f"📄 Loading pair data from {cache_path}...")
    return _read_csv_any(cache_path)

def split_data(df, train_ratio=0.6, test_ratio=0.2):
    """
    Chronological split into Train, Test, Validation
    avoiding look-ahead bias.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    test_end = train_end + int(n * test_ratio)

    train = df.iloc[:train_end].copy()
    test = df.iloc[train_end:test_end].copy()
    valid = df.iloc[test_end:].copy()

    return train, test, valid
