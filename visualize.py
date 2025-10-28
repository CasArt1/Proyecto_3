# visualize.py

import matplotlib.pyplot as plt
import pandas as pd

def plot_results(stock_x, stock_y, result, z_entry, z_exit):
    """
    Plots:
    1) Prices of X and Y
    2) Spread + Z-score bands
    3) Hedge ratio over time
    """

    # ✅ Support both old + new key formats
    spread = result.get("spread") or result.get("Spread")
    z = result.get("z") or result.get("Z")
    hedge = result.get("hedge") or result.get("Hedge")

    spread = pd.Series(spread, index=stock_x.index, name="Spread")
    z = pd.Series(z, index=stock_x.index, name="Z")
    hedge = pd.Series(hedge, index=stock_x.index, name="Hedge Ratio")

    plt.figure(figsize=(14, 10))

    # ------- (1) Prices -------
    plt.subplot(3, 1, 1)
    plt.plot(stock_x.index, stock_x, label=stock_x.name, color="blue", linewidth=1)
    plt.plot(stock_y.index, stock_y, label=stock_y.name, color="red", linewidth=1)
    plt.title("Asset Prices")
    plt.legend()
    plt.grid(True)

    # ------- (2) Spread & Z-score bands -------
    plt.subplot(3, 1, 2)
    plt.plot(spread.index, spread, label="Spread", linewidth=1)
    plt.axhline(z_entry, color="green", linestyle="--", label=f"Entry Z ({z_entry})")
    plt.axhline(-z_entry, color="green", linestyle="--")
    plt.axhline(z_exit, color="red", linestyle="--", label=f"Exit Z ({z_exit})")
    plt.axhline(-z_exit, color="red", linestyle="--")
    plt.title("Spread with Entry/Exit Bands")
    plt.legend()
    plt.grid(True)

    # ------- (3) Hedge Ratio -------
    plt.subplot(3, 1, 3)
    plt.plot(hedge.index, hedge, label="Hedge Ratio β", color="black", linewidth=0.8)
    plt.title("Hedge Ratio (Kalman Filter)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

