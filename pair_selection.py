# ==========================================================
# pair_selection.py
# ==========================================================
# Identifies correlated, cointegrated, and mean-reverting pairs
# using correlation, Engle-Granger, Johansen, and ADF tests.
# ==========================================================

import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def engle_granger(x, y):
    _, p, _ = coint(x, y)
    return p

def johansen_test(x, y):
    df = pd.concat([x, y], axis=1)
    result = coint_johansen(df, det_order=0, k_ar_diff=1)
    return result.lr1[0], result.cvt[0, 1]

def adf_spread(x, y):
    hedge_ratio = np.polyfit(x, y, 1)[0]
    spread = y - hedge_ratio * x
    return adfuller(spread)[1]

def select_pairs(data, min_corr=0.7):
    """
    Scans all ticker pairs and filters those that meet:
      - Correlation >= min_corr
      - Engle-Granger p < 0.05
      - Johansen stat > crit
      - ADF(spread) p < 0.05
    Returns a sorted DataFrame of valid pairs.
    """
    results = []
    tickers = data.columns
    print("🔍 Evaluating possible pairs...\n")

    for s1, s2 in tqdm(combinations(tickers, 2)):
        x, y = np.log(data[s1].dropna()), np.log(data[s2].dropna())
        if len(x) != len(y):  # align timeframes
            df = pd.concat([x, y], axis=1).dropna()
            x, y = df.iloc[:, 0], df.iloc[:, 1]
        corr = x.corr(y)
        if corr < min_corr:
            continue

        try:
            eg_p = engle_granger(x, y)
            j_stat, j_crit = johansen_test(x, y)
            adf_p = adf_spread(x, y)

            if eg_p < 0.05 and j_stat > j_crit and adf_p < 0.05:
                results.append({
                    "Pair": f"{s1}-{s2}",
                    "Corr": corr,
                    "Engle_p": eg_p,
                    "Johansen_stat": j_stat,
                    "Johansen_crit": j_crit,
                    "ADF_spread_p": adf_p
                })
        except Exception:
            continue

    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("❌ No valid cointegrated + mean-reverting pairs found.")

    df = df.sort_values(by=["Corr", "Engle_p", "ADF_spread_p"], ascending=[False, True, True])
    return df
