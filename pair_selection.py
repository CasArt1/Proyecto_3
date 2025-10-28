# pair_selection.py — Global Tech (~50) with 15y filter + rolling-window cointegration
import os
import datetime as dt
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller

warnings.filterwarnings("ignore")

YEARS = 15
ROLL_YEARS = 3
STEP_MONTHS = 3
MIN_PASS_RATIO = 0.70
MIN_CORR = 0.70
DATA_DIR = "data"
OUT_CSV = os.path.join(DATA_DIR, "stocks.csv")

# Approximate global large/mega cap tech universe (US + ADR + key non-US listings).
# Yahoo tickers often need suffixes (e.g., .HK, .KS, .T, .VX). We rely on the 15y filter anyway.
GLOBAL_TECH_CANDIDATES = [
    # US megacaps / large caps
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","AVGO","ORCL","IBM","CSCO","ADBE","CRM",
    "AMD","INTC","QCOM","TXN","ADI","MU","AMAT","LRCX","KLAC","CDNS","SNPS","INTU","NOW",
    "PYPL","ADP","PAYX","PANW","FTNT","CRWD","ACN","SHOP","SQ","UBER","TSLA",  # (TSLA behaves like Tech)
    # ADRs / non-US listings with long history (most have 15+ years)
    "TSM",               # Taiwan Semi (ADR)
    "ASML.AS","ASML",    # ASML (Euronext / NY ADR)
    "SAP","SAP.DE",      # SAP (US ADR / Xetra)
    "SONY","6758.T",     # Sony (ADR / Tokyo)
    "ERIC","ERIC-B.ST",  # Ericsson (ADR / Stockholm B)
    "NTES",              # NetEase (ADR, long history)
    "INFY","TCS.NS","WIT","HCLTECH.NS","TECHM.NS",  # India IT (ADR / NSE)
    "NXPI","STM","SWKS","MPWR","MRVL","MCHP","ON",   # global semi names (US-listed)
    "ARM",               # note: may be <15y; filter will drop if insufficient history
    "BIDU","BABA","PDD","JD",  # China ADRs (some <15y; filter will drop)
    "0700.HK",           # Tencent (HK)
    "3690.HK",           # Meituan (may be <15y; filter will drop)
    "005930.KS",         # Samsung Elec (KRX)
    "000660.KS",         # SK Hynix (KRX)
    "2330.TW",           # Taiwan Semi (local)
    "0703.HK",           # hypothetical; harmless b/c of filter if invalid
]

def download_one(ticker: str, start, end):
    """Robust per-ticker download (auto_adjust=True). Returns pd.Series Close or None."""
    try:
        df = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False, threads=False
        )
        if df is None or df.empty or "Close" not in df.columns:
            return None
        s = df["Close"].rename(ticker)
        s = s[~s.index.duplicated(keep="first")].dropna()
        return s
    except Exception:
        return None

def filter_by_history(tickers, start, end):
    """Return (keep_list, series_map) for tickers with ≥15y history and enough obs."""
    cutoff = pd.Timestamp(start)
    keep, series = [], {}
    for t in tickers:
        s = download_one(t, start, end)
        if s is None or s.empty:
            continue
        # require first date <= cutoff and at least ~200 trading days/yr
        if s.index.min() <= cutoff and s.count() >= YEARS * 200:
            keep.append(t)
            series[t] = s
    return keep, series

def rolling_windows_index(idx, win_days, step_days):
    start = idx.min()
    end = idx.max()
    anchors = []
    cur = start
    while cur + pd.Timedelta(days=win_days) <= end:
        anchors.append(cur)
        cur += pd.Timedelta(days=step_days)
    locs = []
    for a in anchors:
        i0 = idx.get_indexer([a], method="nearest")[0]
        i1 = idx.get_indexer([a + pd.Timedelta(days=win_days)], method="nearest")[0]
        if i1 > i0 + 5:
            locs.append((i0, i1))
    return locs

def window_tests(x, y):
    """Return (Engle–Granger p, spread ADF p)."""
    eg_p = coint(x, y)[1]
    try:
        beta = np.polyfit(y, x, 1)[0]
        spread = x - beta * y
    except Exception:
        spread = x - y
    try:
        adf_p = adfuller(spread, maxlag=1, autolag="AIC")[1]
    except Exception:
        adf_p = 1.0
    return float(eg_p), float(adf_p)

def score_pair(x, y, idx, win_days, step_days):
    corr = pd.Series(x, index=idx).corr(pd.Series(y, index=idx))
    if not np.isfinite(corr) or corr < MIN_CORR:
        return None
    wins = rolling_windows_index(idx, win_days, step_days)
    if len(wins) == 0:
        return None

    eg_ps, adf_ps, pass_count = [], [], 0
    for i0, i1 in wins:
        xx, yy = x[i0:i1], y[i0:i1]
        if len(xx) < 50 or len(yy) < 50:
            continue
        eg_p, adf_p = window_tests(xx, yy)
        eg_ps.append(eg_p); adf_ps.append(adf_p)
        if eg_p < 0.05 and adf_p < 0.05:
            pass_count += 1

    if len(eg_ps) == 0:
        return None

    pass_ratio = pass_count / len(eg_ps)
    if pass_ratio < MIN_PASS_RATIO:
        return None

    eg_mean = float(np.mean(eg_ps))
    adf_mean = float(np.mean(adf_ps))
    eg_score = 1.0 - eg_mean
    adf_score = 1.0 - adf_mean
    score = float(corr) * (0.5 * eg_score + 0.5 * adf_score) * pass_ratio

    return {
        "corr": float(corr),
        "eg_mean": eg_mean,
        "adf_mean": adf_mean,
        "pass_ratio": float(pass_ratio),
        "score": float(score),
    }

def find_best_pair(closes: pd.DataFrame | None = None, years: int = YEARS):
    """
    Build a global tech universe (~50 large/mega-cap),
    keep only tickers with >= 15y data, run rolling-window cointegration,
    save the BEST pair's prices to data/stocks.csv, and return (x, y).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)

    if closes is None:
        candidates = list(dict.fromkeys(GLOBAL_TECH_CANDIDATES))  # de-dup while keep order
        keep, series_map = filter_by_history(candidates, start=start, end=end)
        if len(keep) < 2:
            raise RuntimeError("No global tech tickers with ≥15y history found.")
        df = pd.concat([series_map[t] for t in keep], axis=1).dropna()
        df = df.asfreq("B").ffill()
    else:
        df = closes.copy().dropna().asfreq("B").ffill()

    print(f"Universe (global tech, ≥15y): {df.shape[1]} tickers")

    idx = df.index
    win_days = int(ROLL_YEARS * 252)
    step_days = int(STEP_MONTHS * 21)

    best, best_pair = None, None
    cols = df.columns.tolist()
    for a, b in combinations(cols, 2):
        x = df[a].values
        y = df[b].values
        res = score_pair(x, y, idx, win_days, step_days)
        if res is None:
            continue
        if (best is None) or (res["score"] > best["score"]):
            best = res
            best_pair = (a, b)

    if best_pair is None:
        raise RuntimeError("No pair met the rolling-window cointegration criteria.")

    best_df = df.loc[:, [best_pair[0], best_pair[1]]].copy()
    best_df.to_csv(OUT_CSV)
    print(f"🏆 Best pair selected: {best_pair}")
    print(f"💾 Saved best pair {best_pair} → {OUT_CSV}")
    print(f"Rolling stats: corr={best['corr']:.3f}, pass_ratio={best['pass_ratio']:.2f}, "
          f"EG_mean_p={best['eg_mean']:.4f}, ADF_mean_p={best['adf_mean']:.4f}, score={best['score']:.4f}")
    return best_pair
