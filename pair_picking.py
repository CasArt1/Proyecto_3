# ==========================================================
# 1. IMPORTS
# ==========================================================
import os
import datetime as dt
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from yahooquery import Ticker
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# ==========================================================
# 2. PARAMETERS
# ==========================================================
years = 15
tickers = [
    "AAPL","MSFT","GOOGL","AMZN","META","IBM","ORCL","CSCO",
    "NVDA","INTC","AMD","TXN","QCOM","AVGO","ADI","AMAT","LRCX","KLAC","MU",
    "ADBE","CRM","INTU","CTSH","HPQ",
    "DELL","HPE","NTAP","XRX","STX","WDC",
    "ACN","PAYX","ADP","CDNS","SNPS",
    "FFIV","CHKP","PANW","EBAY","NFLX","EXPE","TRIP"
]

# ==========================================================
# 3. DOWNLOAD DATA (Yahooquery)
# ==========================================================
start = (dt.datetime.now() - dt.timedelta(days=years*365)).strftime("%Y-%m-%d")
end   = dt.datetime.now().strftime("%Y-%m-%d")

print("⬇️ Downloading data via yahooquery...")
tq = Ticker(tickers)
hist = tq.history(start=start, end=end)

# Pivot into wide format
if isinstance(hist.index, pd.MultiIndex):
    closes = hist.reset_index().pivot(index="date", columns="symbol", values="close")
else:
    raise RuntimeError("Unexpected data format")

# Filter valid tickers
min_obs = years * 200
valid = [t for t in closes.columns if closes[t].count() >= min_obs]
closes = closes[valid].dropna()
print(f"✅ Using {len(valid)} tickers with ≥15y data.\n")

# ==========================================================
# 4. TEST FUNCTIONS
# ==========================================================
def engle_granger(x, y):
    _, p, _ = coint(x, y)
    return p

def johansen_test(x, y):
    df = pd.concat([x, y], axis=1)
    result = coint_johansen(df, det_order=0, k_ar_diff=1)
    return result.lr1[0], result.cvt[0, 1]

# ==========================================================
# 5. TEST ALL PAIRS AND FIND BEST ONE
# ==========================================================
pairs = combinations(valid, 2)
results = []

print("🔍 Evaluating all pairs...\n")
for s1, s2 in tqdm(pairs):
    try:
        x, y = np.log(closes[s1]), np.log(closes[s2])
        corr = x.corr(y)
        if corr < 0.7:
            continue
        pval = engle_granger(x, y)
        j_stat, j_crit = johansen_test(x, y)
        if pval < 0.05 and j_stat > j_crit:
            results.append({
                "Pair": f"{s1}-{s2}",
                "Corr": corr,
                "pValue": pval,
                "Johansen_stat": j_stat,
                "Johansen_crit": j_crit
            })
    except Exception:
        continue

df = pd.DataFrame(results)
if df.empty:
    raise RuntimeError("❌ No cointegrated pairs found.")
df = df.sort_values(by=["Corr", "pValue"], ascending=[False, True])
best_pair = df.iloc[0]["Pair"].split("-")
print("\n🏆 Best pair found:", best_pair)

# ==========================================================
# 6. SAVE BEST PAIR TO CSV
# ==========================================================
os.makedirs("data", exist_ok=True)  # create folder if not present
best_df = closes[list(best_pair)].copy()
best_df.to_csv("data/stocks.csv")

print(f"💾 Saved best pair ({best_pair[0]} & {best_pair[1]}) to data/stocks.csv")
print("Data shape:", best_df.shape)
