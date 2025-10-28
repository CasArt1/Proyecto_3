# backtest.py
import numpy as np
import pandas as pd
from kalman_filter import run_kalman


def run_backtest(
    stock_x: pd.Series,
    stock_y: pd.Series,
    q: float = 1e-3,
    r: float = 1e-3,
    entry_z: float = None,
    exit_z: float = None,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    sizing: float = 0.40,
    costs_bps: float = 12.5,
    borrow_annual: float = 0.0025,
) -> dict:
    """
    Pairs backtest with dynamic hedge ratio from run_kalman().

    Signals (spread = x - beta*y in log space):
      - z > +entry  => short spread (short x, long y)
      - z < -entry  => long  spread (long x, short y)
      - |z| < exit  => flat

    sizing = fraction of equity deployed PER LEG (e.g., 0.40 => 40% on x and 40% on y).
    """

    # Allow legacy arg names (z_entry / z_exit)
    if entry_z is None and z_entry is not None:
        entry_z = z_entry
    if exit_z is None and z_exit is not None:
        exit_z = z_exit

    # Assemble and sanity-check data
    df = pd.concat([stock_x.rename("x"), stock_y.rename("y")], axis=1).dropna()
    if len(df) < 200:
        # Not enough data — return empty but well-formed result
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
            "equity_curve": pd.Series(index=df.index, dtype=float),
            "entries": [],
            "exits": [],
            "entry_sides": [],
        }

    # Run Kalman to get hedge ratio, spread, zscore
    kf = run_kalman(df["x"], df["y"], q=q, r=r)
    hedge = kf["hedge_ratio"]
    spread = kf["spread"]
    z = kf["zscore"]

    # Align everything
    keep = spread.notna() & z.notna() & hedge.notna()
    df = df.loc[keep].copy()
    spread = spread.loc[keep]
    z = z.loc[keep]
    hedge = hedge.loc[keep]

    # ===============================
    #   POSITION LOGIC (Sequential)
    # ===============================
    # pos: -1 = short spread, +1 = long spread, 0 = flat
    pos = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        prev = pos.iloc[i - 1]
        zi = z.iloc[i]
        cur = prev

        if prev == 0:
            # Flat → open only on fresh entry signals
            if zi > entry_z:
                cur = -1
            elif zi < -entry_z:
                cur = +1
        else:
            # In a trade → close only when inside exit band
            if abs(zi) < exit_z:
                cur = 0

        pos.iloc[i] = cur

    # ===============================
    #      PnL / COSTS SIMULATION
    # ===============================
    equity = 1.0
    equity_series = [equity]
    daily_rets = []

    bps = costs_bps / 10000.0
    borrow_daily = borrow_annual / 252.0

    trades = 0
    roundtrip_wins = 0

    # Trade markers for plotting
    entries_idx = []
    entry_sides = []  # +1 long-spread, -1 short-spread
    exits_idx = []

    prev_x = df["x"].iloc[0]
    prev_y = df["y"].iloc[0]

    for t in range(1, len(df)):
        px = df["x"].iloc[t]
        py = df["y"].iloc[t]
        beta = hedge.iloc[t]
        p = pos.iloc[t]
        p_prev = pos.iloc[t - 1]

        # Commission on ANY position change (enter, flip, or exit)
        if p != p_prev:
            trades += 1
            equity *= (1 - bps) ** 2  # two legs

            # Track trade markers
            if p_prev == 0 and p != 0:
                entries_idx.append(df.index[t])
                entry_sides.append(int(p))
            elif p_prev != 0 and p == 0:
                exits_idx.append(df.index[t])
                # crude "win" detection: equity higher than on entry (approx)
                # We approximate by checking last step gain/loss on close; for a full
                # accounting you'd store equity-at-entry and compare on exit.
                # Kept simple to avoid state bloat.
                if len(equity_series) >= 2 and equity_series[-1] > equity_series[-2]:
                    roundtrip_wins += 1

        # Notional per leg
        leg_notional = equity * sizing

        # Log returns for numerical stability
        ret_x = np.log(px / prev_x)
        ret_y = np.log(py / prev_y)

        pnl = 0.0
        if p == +1:
            # Long spread: +x, -beta*y
            pnl = leg_notional * ret_x - (leg_notional * beta) * ret_y
            # Borrow on the short leg (y)
            equity -= (leg_notional * abs(beta)) * borrow_daily
        elif p == -1:
            # Short spread: -x, +beta*y
            pnl = -(leg_notional * ret_x) + (leg_notional * beta) * ret_y
            # Borrow on the short leg (x)
            equity -= leg_notional * borrow_daily

        equity += pnl

        # Record returns/equity
        prev_equity = equity_series[-1]
        daily_rets.append(equity / prev_equity - 1.0)
        equity_series.append(equity)

        prev_x, prev_y = px, py

    equity_series = pd.Series(equity_series, index=df.index)  # same length as df
    rets = pd.Series(daily_rets, index=df.index[1:]).replace([np.inf, -np.inf], 0).fillna(0)

    # ===============================
    #            METRICS
    # ===============================
    final_equity = float(equity_series.iloc[-1])
    total_return = final_equity - 1.0
    sharpe = (rets.mean() / (rets.std() + 1e-12) * np.sqrt(252)) if rets.std() > 0 else 0.0

    peak = equity_series.cummax()
    dd = (equity_series / peak - 1.0).min() if len(peak) else 0.0
    max_dd_pct = float(dd * 100.0)

    win_rate = (roundtrip_wins / trades * 100.0) if trades > 0 else 0.0

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
        # Series for plotting
        "Spread": spread,
        "Z": z,
        "Hedge": hedge,
        "equity_curve": equity_series,
        # Trade markers for your visualize.py
        "entries": entries_idx,
        "exits": exits_idx,
        "entry_sides": entry_sides,
    }