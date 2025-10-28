# ==========================================================
# visualize.py
# ==========================================================
# Plot spread, Z-score, PnL, and hedge ratio evolution.
# ==========================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_results(stock_x, stock_y, result, entry_z, exit_z, title=None):
    """
    Visualizes:
    1️⃣ Stock prices and spread
    2️⃣ Z-score with entry/exit thresholds
    3️⃣ Cumulative PnL curve
    4️⃣ Hedge ratio (beta) evolution
    """
    spread = result["Spread"]
    z = result["Zscore"]
    cum_pnl = result["Cumulative"]
    betas = result["Betas"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # 1️⃣ Stock Prices and Spread
    axs[0].plot(stock_x, label=stock_x.name, alpha=0.8)
    axs[0].plot(stock_y, label=stock_y.name, alpha=0.8)
    axs[0].set_title("Asset Prices", fontsize=13)
    axs[0].legend()

    # 2️⃣ Spread with Z-score levels
    axs[1].plot(spread, label="Spread", color="blue", alpha=0.8)
    axs[1].axhline(spread.mean(), color="black", linestyle="--", alpha=0.6)
    axs[1].set_title("Spread Evolution", fontsize=13)
    axs[1].legend()

    # 3️⃣ Z-score and thresholds
    axs[2].plot(z, color="purple", label="Z-score")
    axs[2].axhline(entry_z, color="red", linestyle="--", label="Entry +Z")
    axs[2].axhline(-entry_z, color="green", linestyle="--", label="Entry -Z")
    axs[2].axhline(exit_z, color="orange", linestyle=":")
    axs[2].axhline(-exit_z, color="orange", linestyle=":")
    axs[2].axhline(0, color="black", linestyle="--", linewidth=1)
    axs[2].set_title("Z-score Trading Signals", fontsize=13)
    axs[2].legend()

    # 4️⃣ Cumulative PnL
    axs[3].plot(cum_pnl, color="darkgreen")
    axs[3].axhline(0, color="black", linestyle="--")
    axs[3].set_title("Cumulative PnL", fontsize=13)

    plt.suptitle(title or f"Pairs Trading Backtest ({stock_x.name} & {stock_y.name})", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_hedge_ratio(betas, title="Dynamic Hedge Ratio (β) Evolution"):
    """Plots Kalman Filter hedge ratio over time."""
    plt.figure(figsize=(10, 4))
    plt.plot(betas, color="teal")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("β (Hedge Ratio)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
