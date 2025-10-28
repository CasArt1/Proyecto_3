import numpy as np
from kalman_filter import run_kalman


def run_backtest(x, y,
                 q=0.001, r=0.001,
                 z_entry=2.0, z_exit=0.5,
                 sizing=0.40,
                 costs_bps=12.5,
                 borrow_annual=0.0025):

    # Reset index for safety
    df = (
        {'x': x.values, 'y': y.values}
    )
    x = np.array(x)
    y = np.array(y)

    # === Kalman Filter Estimation === #
    kf = run_kalman(x, y, q=q, r=r)

    beta = np.array(kf["beta"])
    alpha = np.array(kf["alpha"])
    spread = np.array(kf["spread"])
    zscore = np.array(kf["zscore"])

    # === Trading State === #
    position = 0  # 0=flat, 1=long spread, -1=short spread
    pnl = [1.0]  # Equity curve
    trades = 0

    # Convert fees
    cost_rate = costs_bps / 10000
    borrow_daily = borrow_annual / 252

    # === Backtest Loop === #
    for i in range(1, len(x)):
        equity = pnl[-1]

        if position == 0:
            # OPEN TRADE
            if zscore[i] > z_entry:  # Spread too high ✅ SHORT spread
                position = -1
                trades += 1
                equity *= (1 - cost_rate)

            elif zscore[i] < -z_entry:  # Spread too low ✅ LONG spread
                position = 1
                trades += 1
                equity *= (1 - cost_rate)

        else:
            # CLOSE TRADE
            if abs(zscore[i]) < z_exit:
                position = 0
                equity *= (1 - cost_rate)  # exit cost

        # === Apply returns from position === #
        leg_allocation = sizing * equity  # 40% each leg
        hedge_ratio = beta[i]

        if position != 0:
            # Spread return approximation
            ret = (x[i] - x[i - 1]) - hedge_ratio * (y[i] - y[i - 1])

            equity += position * leg_allocation * ret

            # Borrow cost applies if short leg exists
            if position != 0:
                equity -= abs(leg_allocation * borrow_daily)

        pnl.append(equity)

    pnl = np.array(pnl)

    total_return = (pnl[-1] / pnl[0] - 1) * 100
    daily_returns = np.diff(pnl) / pnl[:-1]
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    max_dd = (np.min(pnl) / np.max(pnl) - 1) * 100
    win_rate = np.mean(daily_returns > 0) * 100

    return {
        # Performance metrics
        "final_equity": float(pnl[-1]),
        "total_return_pct": float(total_return),
        "sharpe_daily": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "trades": trades,
        "win_rate_pct": float(win_rate),
        "sizing_mode": "per_leg",
        "costs_bps": costs_bps,
        "borrow_annual_pct": borrow_annual * 100,

        # ✅ Add these for plotting
        "spread": spread,
        "zscore": zscore,
        "beta": beta,
        "equity_curve": pnl
    }

