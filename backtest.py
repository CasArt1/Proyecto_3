# ==========================================================
# backtest.py
# ==========================================================
# Performs mean-reversion backtest based on Kalman-filtered
# hedge ratio and Z-score signal thresholds.
# ==========================================================

import numpy as np
import pandas as pd

def compute_zscore(spread, window=60):
    """Computes rolling Z-score of a spread."""
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std

def run_backtest(stock_x, stock_y, q=0.001, r=0.001, 
                 entry_z=2.0, exit_z=0.5, window=60, kalman_cls=None):
    """
    Runs the Kalman-based pair trading strategy.
    Returns a dictionary with Sharpe ratio, PnL, and trade stats.
    """
    from kalman_filter import KalmanFilterHedgeRatio

    n = len(stock_x)
    kf = kalman_cls(q=q, r=r) if kalman_cls else KalmanFilterHedgeRatio(q=q, r=r)
    betas, spreads = [], []

    # 1️⃣ Apply Kalman filter dynamically
    for t in range(n):
        kf.predict()
        beta_t = kf.update(stock_x.iloc[t], stock_y.iloc[t])
        spread_t = stock_y.iloc[t] - beta_t * stock_x.iloc[t]
        betas.append(beta_t)
        spreads.append(spread_t)

    spread_series = pd.Series(spreads, index=stock_x.index)
    z = compute_zscore(spread_series, window)
    betas = pd.Series(betas, index=stock_x.index)

    # 2️⃣ Generate trading signals
    position = 0  # 1 = long spread, -1 = short spread, 0 = flat
    pnl = []
    for i in range(1, len(z)):
        if position == 0:
            if z.iloc[i] > entry_z:
                position = -1  # short spread
            elif z.iloc[i] < -entry_z:
                position = 1   # long spread
        elif position == 1 and z.iloc[i] > -exit_z:
            position = 0
        elif position == -1 and z.iloc[i] < exit_z:
            position = 0

        # Δspread between steps → profit for position
        pnl.append(position * (spread_series.iloc[i] - spread_series.iloc[i-1]))

    pnl = pd.Series(pnl, index=stock_x.index[1:])
    pnl = pnl.fillna(0)
    sharpe = np.sqrt(252) * pnl.mean() / pnl.std() if pnl.std() > 0 else 0.0
    cumulative = pnl.cumsum()

    return {
        "Sharpe": sharpe,
        "PnL": pnl,
        "Cumulative": cumulative,
        "Betas": betas,
        "Spread": spread_series,
        "Zscore": z
    }
