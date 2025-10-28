# ==========================================================
# main.py
# ==========================================================
# Main execution pipeline for Proyecto3_Enhanced
# ==========================================================

import os
from data_loader import download_data
from pair_selection import select_pairs
from optimize import optimize_parameters
from backtest import run_backtest
from kalman_filter import KalmanFilterHedgeRatio

# ----------------------------------------------------------
# 1️⃣ Define the tickers you want to analyze
# ----------------------------------------------------------
tickers = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    "META",  # Meta Platforms
    "IBM",   # IBM
    "ORCL",  # Oracle
    "CSCO",  # Cisco
    "NVDA",  # Nvidia
    "INTC",  # Intel
    "AMD",   # AMD
    "TXN",   # Texas Instruments
    "QCOM",  # Qualcomm
    "AVGO",  # Broadcom
    "ADI",   # Analog Devices
    "AMAT",  # Applied Materials
    "LRCX",  # Lam Research
    "KLAC",  # KLA Corporation
    "MU",    # Micron Technology
    "ADBE",  # Adobe
    "CRM",   # Salesforce
    "INTU",  # Intuit
    "CTSH",  # Cognizant
    "HPQ",   # HP Inc.
    "DELL",  # Dell
    "HPE",   # Hewlett Packard Enterprise
    "NTAP",  # NetApp
    "XRX",   # Xerox
    "STX",   # Seagate Technology
    "WDC",   # Western Digital
    "ACN",   # Accenture
    "PAYX",  # Paychex
    "ADP",   # Automatic Data Processing
    "CDNS",  # Cadence Design Systems
    "SNPS",  # Synopsys
    "FFIV",  # F5 Networks
    "CHKP",  # Check Point Software
    "PANW",  # Palo Alto Networks
    "EBAY",  # eBay
    "NFLX",  # Netflix
    "EXPE",  # Expedia
    "TRIP"   # TripAdvisor
]

# ----------------------------------------------------------
# 2️⃣ Download data (15 years of Adjusted Close)
# ----------------------------------------------------------
data = download_data(tickers, years=15)

# ----------------------------------------------------------
# 3️⃣ Select valid pairs (cointegrated + mean-reverting)
# ----------------------------------------------------------
pairs_df = select_pairs(data)
print("\nTop 5 valid pairs:\n", pairs_df.head())

best_pair = pairs_df.iloc[0]["Pair"].split("-")
s1, s2 = best_pair
print(f"\n🏆 Selected best pair: {s1} & {s2}")

# ----------------------------------------------------------
# 4️⃣ Optimize Kalman + strategy parameters
# ----------------------------------------------------------
stock_x, stock_y = data[s1].dropna(), data[s2].dropna()
best_params, best_sharpe = optimize_parameters(stock_x, stock_y, n_trials=50)

# ----------------------------------------------------------
# 5️⃣ Run backtest with best parameters
# ----------------------------------------------------------
final_result = run_backtest(
    stock_x, stock_y,
    q=best_params["q"],
    r=best_params["r"],
    entry_z=best_params["entry_z"],
    exit_z=best_params["exit_z"],
    kalman_cls=KalmanFilterHedgeRatio
)

print(f"\n✅ Final Sharpe Ratio: {final_result['Sharpe']:.4f}")
print(f"📊 Cumulative Return (last day): {final_result['Cumulative'].iloc[-1]:.4f}")

# ----------------------------------------------------------
# 6️⃣ Save final data for record
# ----------------------------------------------------------
os.makedirs("data", exist_ok=True)
best_df = data[[s1, s2]].copy()
best_df.to_csv("data/stocks.csv")
print(f"💾 Saved best pair data to data/stocks.csv")

# ----------------------------------------------------------
# 7️⃣ Visualization
# ----------------------------------------------------------
from visualize import plot_results, plot_hedge_ratio

plot_results(
    stock_x, stock_y,
    result=final_result,
    entry_z=best_params["entry_z"],
    exit_z=best_params["exit_z"],
    title=f"Optimized Pair: {s1}-{s2} | Sharpe: {final_result['Sharpe']:.2f}"
)

plot_hedge_ratio(final_result["Betas"])
