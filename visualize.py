import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_results(stock_x, stock_y, result, z_entry, z_exit):

    # Convert numpy arrays to pandas Series with proper index
    spread = pd.Series(result["spread"], index=stock_x.index, name="Spread")
    zscore = pd.Series(result["zscore"], index=stock_x.index, name="Z-Score")
    beta   = pd.Series(result["beta"],   index=stock_x.index, name="Beta")
    equity = pd.Series(result["equity_curve"], index=stock_x.index, name="Equity")

    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # ====================================
    # 1️⃣ Price relationship + hedge ratio
    # ====================================
    axs[0].plot(stock_x.index, stock_x, label=stock_x.name, alpha=0.8)
    axs[0].plot(stock_y.index, stock_y * beta, label=f"{stock_y.name} × beta", alpha=0.8)
    axs[0].set_title("Price Relationship & Hedge Ratio")
    axs[0].legend()

    # ====================================
    # 2️⃣ Spread + Z-score signals
    # ====================================
    axs[1].plot(spread.index, spread, label="Spread", alpha=0.9)
    axs[1].axhline(z_entry, color='red', linestyle="--", label="Entry ±Z")
    axs[1].axhline(-z_entry, color='red', linestyle="--")
    axs[1].axhline(z_exit, color='green', linestyle="--", label="Exit")
    axs[1].axhline(-z_exit, color='green', linestyle="--")
    axs[1].set_title("Spread with Z-Score Levels")
    axs[1].legend()

    # ====================================
    # 3️⃣ Equity curve
    # ====================================
    axs[2].plot(equity.index, equity, color="blue", label="Equity Curve")
    axs[2].set_title("Strategy Equity Curve")
    axs[2].legend()

    plt.tight_layout()
    plt.show()
