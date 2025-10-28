# backtest.py
import numpy as np
import pandas as pd
from kalman_filter import run_kalman

def run_backtest(stock_x: pd.Series,
                 stock_y: pd.Series,
                 q: float = 1e-3,
                 r: float = 1e-3,
                 entry_z: float = None,
                 exit_z: float = None,
                 z_entry: float = 2.0,
                 z_exit: float = 0.5,
                 sizing: float = 0.40,
                 costs_bps: float = 12.5,
                 borrow_annual: float = 0.0025) -> dict:
    """
    Pairs backtest with dynamic hedge ratio from run_kalman().
    Signals:
      - z > +entry => short spread (short x, long y)
      - z < -entry => long  spread (long x, short y)
      - |z| < exit => flat
    sizing = fraction of equity used per leg (e.g., 0.40 => 40% on x and 40% on y)
    """
    if entry_z is None and z_entry is not None:
        entry_z = z_entry
    if exit_z is None and z_exit is not None:
        exit_z = z_exit

    df = pd.concat([stock_x.rename("x"), stock_y.rename("y")], axis=1).dropna()
    if len(df) < 200:
        return {
            "final_equity": 1.0,
            "total_return_pct": 0.0,
            "sharpe_daily": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "sizing_mode": "per_leg",
            "costs_bps": costs_bps,
            "borrow_annual_pct": borrow_annual * 100.0,
            "Spread": pd.Series(index=df.index, dtype=float),
            "Z": pd.Series(index=df.index, dtype=float),
            "Hedge": pd.Series(index=df.index, dtype=float),
        }

    # KF/rolling-OLS info
    kf = run_kalman(df["x"], df["y"], q=q, r=r)
    hedge = kf["hedge_ratio"]
    spread = kf["spread"]
    z = kf["zscore"]

    # Align & clean
    keep = spread.notna() & z.notna() & hedge.notna()
    df = df.loc[keep].copy()
    spread = spread.loc[keep]
    z = z.loc[keep]
    hedge = hedge.loc[keep]

    # Positions: -1 short spread, +1 long spread, 0 flat
    pos = pd.Series(0, index=df.index, dtype=int)
    pos[z > +entry_z] = -1
    pos[z < -entry_z] = +1
    # Exit when |z| < exit
    pos[(z.abs() < exit_z)] = 0
    # Hold until exit (carry forward position)
    pos = pos.replace(to_replace=0, method="ffill").fillna(0).astype(int)

    # CASH model — use % of equity per leg each day
    equity = 1.0
    equity_series = []
    daily_rets = []
    prev_pos = 0
    prev_x = df["x"].iloc[0]
    prev_y = df["y"].iloc[0]
    annual_borrow_daily = borrow_annual / 252.0
    bps = costs_bps / 10000.0

    trades = 0
    wins = 0

    for t in range(1, len(df)):
        px = df["x"].iloc[t]
        py = df["y"].iloc[t]
        beta = hedge.iloc[t]
        p = pos.iloc[t]
        p_prev = pos.iloc[t-1]

        # position change => commission on both legs
        if p != p_prev:
            trades += 1
            equity *= (1 - bps) ** 2  # entry cost on both legs

        # Notional per leg
        leg_notional = equity * sizing  # each leg
        # PnL (use log returns to be robust)
        ret_x = np.log(px / prev_x)
        ret_y = np.log(py / prev_y)

        # Strategy legs:
        # long spread: +x, -beta*y
        # short spread: -x, +beta*y
        pnl = 0.0
        if p == +1:
            pnl = leg_notional * ret_x - (leg_notional * beta) * ret_y
            # borrow on short y leg
            borrow_notional = leg_notional * abs(beta)
            equity -= borrow_notional * annual_borrow_daily
        elif p == -1:
            pnl = -(leg_notional * ret_x) + (leg_notional * beta) * ret_y
            # borrow on short x leg
            borrow_notional = leg_notional
            equity -= borrow_notional * annual_borrow_daily
        else:
            pnl = 0.0

        equity += pnl

        # Exit -> commission on both legs
        if p != 0 and (abs(z.iloc[t]) < exit_z) and p_prev != 0:
            equity *= (1 - bps) ** 2  # exit cost

        daily_ret = (equity_series[-1] if equity_series else 1.0)
        daily_rets.append(equity / daily_ret - 1.0)
        equity_series.append(equity)

        prev_x, prev_y = px, py

        # Win if equity rose on a round-trip (approx via sign change)
        if p_prev != 0 and p == 0:
            if equity_series[-1] > equity_series[-2]:
                wins += 1

    equity_series = pd.Series(equity_series, index=df.index[1:])
    rets = pd.Series(daily_rets, index=df.index[1:]).replace([np.inf, -np.inf], 0).fillna(0)

    # Metrics
    final_equity = float(equity_series.iloc[-1]) if len(equity_series) else 1.0
    total_return = final_equity - 1.0
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(252) if rets.std() > 0 else 0.0
    peak = equity_series.cummax()
    dd = (equity_series / peak - 1.0).min() if len(peak) else 0.0
    max_dd_pct = dd * 100.0
    win_rate = (wins / trades * 100.0) if trades > 0 else 0.0

    # Provide series for plotting module (expects capitalized keys)
    return {
        "final_equity": final_equity,
        "total_return_pct": total_return * 100.0,
        "sharpe_daily": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "trades": trades,
        "win_rate_pct": win_rate,
        "sizing_mode": "per_leg",
        "costs_bps": costs_bps,
        "borrow_annual_pct": borrow_annual * 100.0,
        "Spread": spread,                # pandas Series
        "Z": z,                          # pandas Series
        "Hedge": hedge,                  # pandas Series
        "equity_curve": equity_series    # ✅ NEW — enables PnL chart
    }


