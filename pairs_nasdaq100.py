# pairs_nasdaq100.py
# Find the best pair among NASDAQ-100 using cointegration + simple mean-reversion backtest

import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Install yfinance and statsmodels if needed:
# pip install yfinance statsmodels tqdm lxml

import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from tqdm import tqdm

# ----------------------------
# 1) Universe (NASDAQ-100)
# ----------------------------
HARDCODED_NDX = [
    # Fallback approximate list (kept short here—extend as desired if scrape fails)
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","COST","TSLA",
    "ADBE","NFLX","AMD","PEP","CSCO","LIN","INTC","CMCSA","TXN","AMAT",
    "QCOM","INTU","HON","AMGN","SBUX","PDD","MU","ADP","BKNG","PLTR",
    "MDLZ","REGN","ISRG","VRTX","ABNB","GILD","PANW","ADI","LRCX","PYPL",
    "CRWD","MRVL","SNPS","MELI","ASML","KLAC","CSX","ORLY","ADSK","CHTR",
    "MRNA","CDNS","CPRT","KDP","AEP","CTAS","NXPI","MAR","MNST","FTNT",
    "PCAR","ODFL","EA","ROST","DDOG","TEAM","WDAY","PAYX","KHC","EXC",
    "VRSK","DXCM","IDXX","XEL","CTSH","FAST","VRSN","AZN","CSGP","GEHC",
    "GFS","ROP","MCHP","LCID","ON","ANSS","ZS","SPLK","BKR","SIRI","CHKP",
    "VERX","DOCU","OKTA","META","ARM","KVYO","SMCI"  # some newer names; duplicates will be dropped
]

def get_nasdaq100_tickers():
    try:
        # Wikipedia table
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        # Find the constituents table (usually first or second)
        candidates = [t for t in tables if "Ticker" in t.columns or "Symbol" in t.columns]
        if not candidates:
            raise ValueError("No constituents table found.")
        t = candidates[0]
        col = "Ticker" if "Ticker" in t.columns else "Symbol"
        tickers = (
            t[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\.", "-", regex=True)  # BRK.B style to BRK-B if ever appears
            .unique()
            .tolist()
        )
        # Remove any blanks/oddities
        tickers = [tk for tk in tickers if tk.isalnum() or "-" in tk]
        return sorted(set(tickers))
    except Exception:
        # Fallback
        return sorted(set(HARDCODED_NDX))

# ----------------------------
# 2) Data download
# ----------------------------
# ----------------------------
# 2) Data download (improved 5-year version)
# ----------------------------
def download_prices(tickers, period="5y", interval="1d"):
    """
    Downloads adjusted close prices for all tickers individually to ensure full 5-year coverage.
    Slower but more reliable than bulk download.
    """
    import yfinance as yf
    import pandas as pd
    import time

    closes = pd.DataFrame()

    print(f"Downloading {len(tickers)} tickers individually ({period}, {interval})...")

    for i, tk in enumerate(tickers, 1):
        try:
            data = yf.download(
                tk,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,  # single-threaded, avoids truncation issues
            )

            if not data.empty:
                closes[tk] = data["Close"]
                print(f"[{i}/{len(tickers)}] {tk}: {len(data)} rows from {data.index.min().date()} to {data.index.max().date()}")
            else:
                print(f"[{i}/{len(tickers)}] ⚠️ No data for {tk}")

            time.sleep(0.3)  # small pause to avoid Yahoo throttling

        except Exception as e:
            print(f"[{i}/{len(tickers)}] ❌ Failed {tk}: {e}")
            continue

    # Clean and align data
    closes = closes.ffill().bfill()
    print(f"\n✅ Finished downloading {closes.shape[1]} tickers with {closes.shape[0]} total days.")
    return closes


# ----------------------------
# 3) Stats helpers
# ----------------------------
def hedge_ratio(y, x):
    # OLS y ~ alpha + beta*x
    X = add_constant(x.values)
    model = OLS(y.values, X).fit()
    beta = model.params[1]
    return beta

def adf_pvalue(series):
    series = pd.Series(series).dropna()
    if series.std() == 0 or len(series) < 30:
        return 1.0  # not enough info
    return adfuller(series, maxlag=1, autolag="AIC")[1]

def halflife_of_mean_reversion(spread):
    # Δs_t = a + b * s_{t-1} + ε_t  => halflife = -ln(2)/b, if b<0
    s = pd.Series(spread).dropna()
    if len(s) < 50:
        return np.nan
    lag = s.shift(1).dropna()
    ds = s.diff().dropna()
    lag = lag.loc[ds.index]
    X = add_constant(lag.values)
    model = OLS(ds.values, X).fit()
    b = model.params[1]
    if b >= 0:
        return np.inf
    return -np.log(2) / b

def zscore(series, window=60):
    m = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - m) / sd

# ----------------------------
# 4) Simple mean-reversion backtest on the spread
#    Rules:
#      - z > +2  => short spread (short y, long beta*x)
#      - z < -2  => long spread (long y, short beta*x)
#      - exit when |z| < 0.5
#      - daily rets dollar-neutral; include 10 bps per leg when position changes
# ----------------------------
def backtest_pair(y, x, beta, zwin=60, enter=2.0, exit=0.5, tc_bps=10):
    # spread = y - beta*x
    spread = y - beta * x
    z = zscore(spread, window=zwin)
    z = z.dropna()
    if len(z) < 100:
        return np.nan, {}

    # Position: +1 = long spread, -1 = short spread, 0 = flat
    pos = pd.Series(0, index=z.index, dtype=float)

    # Signals
    pos[z > enter] = -1.0
    pos[z < -enter] = +1.0
    # Exit
    pos[(z.abs() < exit)] = 0.0

    # Carry forward positions until exit
    pos = pos.replace(to_replace=0.0, method="ffill").fillna(0.0)
    pos[(z.abs() < exit)] = 0.0  # ensure exits
    pos = pos.replace(to_replace=0.0, method="ffill").fillna(0.0)

    # Returns of spread: r_spread = r_y - beta * r_x
    y_ret = y.pct_change().reindex(z.index).fillna(0.0)
    x_ret = x.pct_change().reindex(z.index).fillna(0.0)
    spread_ret = y_ret - beta * x_ret

    # Strategy returns: pos * spread_ret
    strat_ret = pos * spread_ret

    # Transaction costs when position changes (two legs): 2 * tc_bps
    # Apply on absolute change in position
    pos_change = pos.diff().abs().fillna(0.0)
    tc = pos_change * (2 * tc_bps / 10000.0)
    strat_ret_net = strat_ret - tc

    if strat_ret_net.std() == 0 or strat_ret_net.isna().all():
        sharpe = np.nan
    else:
        sharpe = np.sqrt(252) * strat_ret_net.mean() / strat_ret_net.std()

    metrics = {
        "sharpe": sharpe,
        "ann_ret_%": 252 * strat_ret_net.mean() * 100,
        "ann_vol_%": np.sqrt(252) * strat_ret_net.std() * 100,
        "trades": int((pos_change > 0).sum()),
    }
    return sharpe, metrics

# ----------------------------
# 5) Main search over pairs
# ----------------------------
def score_all_pairs(closes,
                    max_pairs=None,
                    adf_threshold=0.05,
                    hl_min=2,
                    hl_max=60,
                    zwin=60):
    tickers = closes.columns.tolist()
    pairs = list(itertools.combinations(tickers, 2))
    if max_pairs:
        pairs = pairs[:max_pairs]

    results = []

    for a, b in tqdm(pairs, desc="Scanning pairs"):
        y = closes[a].dropna()
        x = closes[b].dropna()
        common = y.index.intersection(x.index)
        if len(common) < 250:
            continue
        y = y.loc[common]
        x = x.loc[common]

        # OLS hedge ratio
        try:
            beta = hedge_ratio(y, x)
        except Exception:
            continue

        spread = y - beta * x
        pval = adf_pvalue(spread)
        if np.isnan(pval) or pval > adf_threshold:
            continue  # not cointegrated enough

        hl = halflife_of_mean_reversion(spread)
        if not np.isfinite(hl) or hl < hl_min or hl > hl_max:
            continue

        sharpe, metrics = backtest_pair(y, x, beta, zwin=zwin)
        if np.isnan(sharpe):
            continue

        results.append({
            "pair": (a, b),
            "beta_y_on_x": beta,
            "adf_pvalue": pval,
            "halflife": hl,
            "sharpe": metrics["sharpe"],
            "ann_ret_%": metrics["ann_ret_%"],
            "ann_vol_%": metrics["ann_vol_%"],
            "trades": metrics["trades"],
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values(["sharpe", "ann_ret_%"], ascending=False).reset_index(drop=True)
    return df

# ----------------------------
# 6) Run
# ----------------------------
if __name__ == "__main__":
    print("Getting NASDAQ-100 tickers...")
    tickers = get_nasdaq100_tickers()
    print(f"Universe size: {len(tickers)} tickers")

    print("Downloading prices (3y, daily)...")
    prices = download_prices(tickers, period="3y", interval="1d")
    print(f"Got {prices.shape[1]} tickers with usable data.")

    # (Optional) filter out obvious penny-ish names
    med_price = prices.median()
    good = med_price[med_price > 5].index.tolist()
    prices = prices[good]

    print("Scoring pairs...")
    df_pairs = score_all_pairs(
        prices,
        max_pairs=None,        # scan all combinations
        adf_threshold=0.05,    # cointegration significance
        hl_min=2,
        hl_max=60,             # mean reversion in ~2–60 days
        zwin=60                # z-score window
    )

    if df_pairs.empty:
        print("No qualifying pairs found. Try loosening filters (higher adf_threshold, wider half-life bounds).")
    else:
        print("\nTop 10 pairs by Sharpe:")
        print(df_pairs.head(10).to_string(index=False))
        best = df_pairs.iloc[0]
        a, b = best["pair"]
        print("\nBest pair to trade:")
        print(f"  {a} vs {b}")
        print(f"  Sharpe: {best['sharpe']:.2f} | ADF p-value: {best['adf_pvalue']:.4f} | Half-life: {best['halflife']:.1f} days")
        print(f"  Hedge ratio (y on x): {best['beta_y_on_x']:.3f} | Trades: {int(best['trades'])}")
