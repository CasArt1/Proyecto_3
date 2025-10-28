# pair_selection.py
import os
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression

# ✅ Final Universe: 50 Tech Stocks
def get_us_tech50() -> list[str]:
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN",
        "IBM", "ORCL", "CSCO", "INTC",
        "NVDA", "AMD", "TXN", "QCOM", "AVGO", "ADI",
        "AMAT", "LRCX", "KLAC", "MU", "MCHP", "ON", "MRVL", "MPWR", "SWKS", "TER",
        "ADBE", "CRM", "INTU", "ADSK", "ANSS", "AKAM", "FTNT", "VRSN",
        "CDNS", "SNPS",
        "ACN", "ADP", "PAYX", "CTSH",
        "NTAP", "STX", "WDC", "HPQ", "XRX",
        "NFLX", "EBAY", "EXPE",
        "FFIV", "CHKP", "JNPR",
        "TYL",
    ]

# Helpers
def engle_granger_p(x, y):
    try:
        _, pval, _ = coint(np.log(x), np.log(y))
        return float(pval)
    except:
        return 1.0

def ols_beta(x, y):
    X = np.log(y).values.reshape(-1, 1)
    yv = np.log(x).values
    model = LinearRegression().fit(X, yv)
    return float(model.coef_[0])

def spread_series(x, y, beta):
    return np.log(x) - beta * np.log(y)

def adf_p(series):
    try:
        return float(adfuller(series.dropna(), autolag="AIC")[1])
    except:
        return 1.0

def half_life(series):
    s = series.dropna()
    if len(s) < 10:
        return np.inf
    ds = s.diff().dropna()
    lag = s.shift(1).dropna()
    y = ds.values
    x = lag.loc[ds.index].values.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    phi = reg.coef_[0]
    if phi >= 0:
        return np.inf
    return float(-np.log(2) / phi)

def find_top_pairs(closes: pd.DataFrame, top_n=5):
    closes = closes.dropna()
    tickers = closes.columns.tolist()
    rows = []

    for x_t, y_t in combinations(tickers, 2):
        x, y = closes[x_t], closes[y_t]

        # Rolling Correlation 1y (min 252)
        roll_corr = x.rolling(252).corr(y).iloc[-1]
        if roll_corr is None or roll_corr < 0.70:
            continue

        eg = engle_granger_p(x, y)
        if eg >= 0.05:
            continue

        beta = ols_beta(x, y)
        spr = spread_series(x, y, beta)
        adf = adf_p(spr)
        if adf >= 0.05:
            continue

        hl = half_life(spr)
        if hl == np.inf or hl > 50:
            continue

        rows.append({
            "x": x_t, "y": y_t,
            "corr": roll_corr,
            "eg_p": eg,
            "adf_p": adf,
            "half_life": hl,
            "beta": beta
        })

    if not rows:
        print("⚠️ No pairs passed filters — relax thresholds!")
        return []

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["corr", "eg_p", "adf_p", "half_life"],
                        ascending=[False, True, True, True])

    top = df.head(top_n)
    print("\n✅ Top Pairs:")
    print(top.to_string(index=False))

    # Save for Optuna
    os.makedirs("data", exist_ok=True)
    top.to_csv("data/top_pairs.csv", index=False)
    print("\n💾 Saved → data/top_pairs.csv")

    return top
