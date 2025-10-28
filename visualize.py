import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_results(result, stock_x, stock_y, title, entry_z=None, exit_z=None):
    """
    Plot Prices, Spread, Z-Score and Equity Curve including trading thresholds and trade markers.
    """

    spread = pd.Series(result.get("Spread", None))
    z = pd.Series(result.get("Z", None))
    hedge = pd.Series(result.get("Hedge", None))
    equity_curve = pd.Series(result.get("equity_curve", None))

    # Build 4 charts
    fig, axs = plt.subplots(4, 1, figsize=(14, 18), sharex=True)

    # === Price Series ===
    axs[0].plot(stock_x.index, stock_x, label=f"{stock_x.name}", linewidth=1.4)
    axs[0].plot(stock_y.index, stock_y, label=f"{stock_y.name}", linewidth=1.4)
    axs[0].set_title(f"{title} — Leg Prices")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # === Spread ===
    if spread is not None and spread.notna().any():
        axs[1].plot(spread.index, spread, label="Spread", color="purple", linewidth=1.4)
        axs[1].set_title("Spread")
        axs[1].legend()
        axs[1].grid(alpha=0.3)

    # === Z-Score + Entry/Exit thresholds ===
    if z is not None and z.notna().any():
        axs[2].plot(z.index, z, label="Z-score", color="blue", linewidth=1.2)

        if entry_z is not None:
            axs[2].axhline(entry_z, color="red", linestyle="--", linewidth=1, label="Entry")
            axs[2].axhline(-entry_z, color="red", linestyle="--", linewidth=1)

        if exit_z is not None:
            axs[2].axhline(exit_z, color="green", linestyle="--", linewidth=1, label="Exit")
            axs[2].axhline(-exit_z, color="green", linestyle="--", linewidth=1)

        axs[2].set_title("Z-score & Trading Thresholds")
        axs[2].legend()
        axs[2].grid(alpha=0.3)

        # === Equity Curve + Trade Markers ✅ ===
        if equity_curve is not None and equity_curve.notna().any():
            ax = axs[3]
            ax.plot(equity_curve.index, equity_curve.values, label="Equity Curve", color="black", linewidth=1.8)

            # === Trade markers desde el resultado del backtester ===
            entries = result.get("entries", [])
            exits   = result.get("exits", [])
            sides   = result.get("entry_sides", [])

            if entries and sides:
                long_idx  = [ts for ts, s in zip(entries, sides) if s == 1]
                short_idx = [ts for ts, s in zip(entries, sides) if s == -1]
                ax.scatter(long_idx,  equity_curve.loc[long_idx],  marker="^", s=70, label="Long Entry")
                ax.scatter(short_idx, equity_curve.loc[short_idx], marker="v", s=70, label="Short Entry")

            if exits:
                ax.scatter(exits, equity_curve.loc[exits], marker="o", s=55, label="Exit")

            ax.set_title("PnL / Equity Curve + Trades (Backtester)")
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
