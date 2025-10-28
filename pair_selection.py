# pair_selection.py
import os
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression

# -------------------------------
# Universe: 50 US Tech tickers (15y+ history)
# -------------------------------
def get_us_tech50() -> list[str]:
    return [
        # Megacaps / platform
        "AAPL","MSFT","GOOGL","AMZN",
        # Legacy enterprise
        "IBM","ORCL","CSCO","INTC",
        # Semis
        "NVDA","AMD","TXN","QCOM","AVGO","ADI",
        "AMAT","LRCX","KLAC","MU","MCHP","ON","MRVL","MPWR","SWKS","TER",
        # Software
        "ADBE","CRM","INTU","ADSK","ANSS","AKAM","FTNT","VRSN",
        # EDA
        "CDNS","SNPS",
        # IT services & payroll
        "ACN","ADP","PAYX","CTSH",
        # Storage / hardware
        "NTAP","STX","WDC","HPQ","XRX",
        # Internet / media / travel
        "NFLX","EBAY","EXPE",
        # Networking / security
        "FFIV","CHKP","JNPR",
        # Gov / ERP
        "TYL",
    ]  # 50

# -------------------------------
# Helpers
# -------------------------------
def engle_granger_p(x: pd.Series, y: pd.Series) -> float:
    """Engle–Granger cointegration test p-value on log prices."""
    _, pval, _ = coint(np.log(x), np.log(y))
    return float(pval)

def ols_beta(x: pd.Series, y: pd.Series) -> float:
    """OLS hedge ratio for spread = log(x) - beta*log(y)."""
    X = np.log(y).values.reshape(-1, 1)
    yv = np.log(x).values
    model = LinearRegression().fit(X, yv)
    return float(model.coef_[0])

def spread_series(x: pd.Series, y: pd.Series, beta: float) -> pd.Series:
    return np.log(x) - beta * np.log(y)

def adf_p(series: pd.Series) -> float:
    """ADF p-value (lower = more stationary)."""
    series = pd.Series(series).dropna()
    return float(adfuller(series, autolag="AIC")[1])

def half_life(series: pd.Series) -> float:
    """
    Estimate mean-reversion half-life (in trading days) of the spread via AR(1).
    """
    s = pd.Series(series).dropna()
    if len(s) < 20:
        return np.inf
    ds = s.diff().dropna()
    lag = s.shift(1).reindex(ds.index)
    X = lag.values.reshape(-1, 1)
    y = ds.values
    try:
        reg = LinearRegression().fit(X, y)
        phi = float(reg.coef_[0])
    except Exception:
        return np.inf
    if phi >= 0:
        return np.inf
    hl = -np.log(2) / phi
    return float(hl) if np.isfinite(hl) and hl > 0 else np.inf

def beta_stability(x: pd.Series, y: pd.Series, lookback: int = 63) -> float:
    """Std of rolling OLS betas (lower = more stable)."""
    if len(x) <= lookback + 20:
        return np.inf
    betas = []
    for i in range(lookback, len(x)):
        try:
            b = ols_beta(x.iloc[i-lookback:i], y.iloc[i-lookback:i])
            betas.append(b)
        except Exception:
            pass
    b = pd.Series(betas)
    return float(b.std()) if len(b) > 10 else np.inf

# -------------------------------
# Main selector
# -------------------------------
def find_best_pair(closes: pd.DataFrame, save_csv: bool = True) -> tuple[str, str]:
    """
    Given a wide DataFrame of Adj Close (Date index, columns=tickers),
    return (x_ticker, y_ticker) that passes robust filters and ranks best.
    Optionally saves best pair price history to data/stocks.csv
    """
    # Clean to common dates & drop NaNs
    closes = closes.sort_index().dropna(how="all", axis=1)
    # Require ~15y * 200 trading days coverage (rough min)
    tickers = [c for c in closes.columns if closes[c].count() >= int(15 * 200)]
    closes = closes[tickers].dropna()

    if len(closes.columns) < 2:
        raise RuntimeError("Not enough valid tickers with ~15y history.")

    # Rolling 1y corr helper
    def one_year_corr(a: pd.Series, b: pd.Series) -> float:
        win = 252
        ab = pd.concat([a, b], axis=1).dropna()
        if len(ab) < win + 5:
            return -1.0
        return float(ab.iloc[:,0].rolling(win).corr(ab.iloc[:,1]).dropna().iloc[-1])

    rows = []
    pairs = list(combinations(closes.columns, 2))

    for xt, yt in pairs:
        xy = pd.concat([closes[xt], closes[yt]], axis=1).dropna()
        if len(xy) < 252*5:
            continue
        x, y = xy.iloc[:,0], xy.iloc[:,1]

        # 1) Rolling corr (1y)
        rc = one_year_corr(x, y)
        if rc < 0.70:
            continue

        # 2) EG cointegration on log prices
        eg = engle_granger_p(x, y)
        if eg >= 0.05:
            continue

        # 3) Spread stationarity
        beta = ols_beta(x, y)
        spr = spread_series(x, y, beta)
        spr_adf = adf_p(spr)
        if spr_adf >= 0.05:
            continue

        # 4) Half-life (≤ 50 trading days)
        hl = half_life(spr)
        if not np.isfinite(hl) or hl > 50:
            continue

        # 5) Beta stability (≤ 0.50)
        drift = beta_stability(x, y, lookback=63)
        if not np.isfinite(drift) or drift > 0.50:
            continue

        rows.append({
            "x": xt, "y": yt,
            "roll_corr_1y": rc,
            "eg_p": eg,
            "spread_adf_p": spr_adf,
            "half_life": hl,
            "beta": beta,
            "beta_drift": drift
        })

    if not rows:
        # Diagnostic printout so you know which filter kills most pairs
        failed = dict(corr=0, eg=0, adf=0, hl=0, drift=0)
        for xt, yt in pairs:
            xy = pd.concat([closes[xt], closes[yt]], axis=1).dropna()
            if len(xy) < 252*5:
                continue
            x, y = xy.iloc[:,0], xy.iloc[:,1]

            rc = one_year_corr(x, y)
            if rc < 0.70:
                failed["corr"] += 1
                continue
            eg = engle_granger_p(x, y)
            if eg >= 0.05:
                failed["eg"] += 1
                continue
            beta = ols_beta(x, y)
            spr = spread_series(x, y, beta)
            spr_adf = adf_p(spr)
            if spr_adf >= 0.05:
                failed["adf"] += 1
                continue
            hl = half_life(spr)
            if not np.isfinite(hl) or hl > 50:
                failed["hl"] += 1
                continue
            drift = beta_stability(x, y, lookback=63)
            if not np.isfinite(drift) or drift > 0.50:
                failed["drift"] += 1
                continue

        print("\n⚠️ FILTER RESULTS — No pair passed all filters:")
        for k, v in failed.items():
            print(f"Failed {k}: {v}")
        raise RuntimeError("No pairs passed all filters. Adjust thresholds slightly.")

    df = pd.DataFrame(rows).sort_values(
        by=["roll_corr_1y", "eg_p", "spread_adf_p", "half_life", "beta_drift"],
        ascending=[False, True, True, True, True]
    )

    print("\nTop candidates (up to 20):")
    print(df.head(20).to_string(index=False))

    best = df.iloc[0]
    x_t, y_t = best["x"], best["y"]

    if save_csv:
        os.makedirs("data", exist_ok=True)
        closes[[x_t, y_t]].to_csv("data/stocks.csv")
        print(f"\n💾 Saved best pair ('{x_t}', '{y_t}') → data/stocks.csv")

    return x_t, y_t
