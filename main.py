# main.py
import pandas as pd
from data_loader import download_data, split_data   # must exist
from pair_selection import get_us_tech50, find_best_pair
from backtest import run_backtest
from visualize import plot_results

def main():
    # -----------------------------
    # 1) Download 15y for 50 US tech
    # -----------------------------
    tickers = get_us_tech50()
    print("⬇️ Downloading 15y Adj Close for 50 US tech…")
    closes = download_data(tickers=tickers, years=15)   # returns wide DF (Date index)
    print(f"✅ Got {len(closes.columns)} tickers, {len(closes):,} rows.")

    # -----------------------------
    # 2) Pick best pair (saves data/stocks.csv with 2 columns)
    # -----------------------------
    x_ticker, y_ticker = find_best_pair(closes, save_csv=True)

    # -----------------------------
    # 3) Load that pair & split T/T/V
    # -----------------------------
    pair_prices = pd.read_csv("data/stocks.csv", index_col=0, parse_dates=True)
    train, test, valid = split_data(pair_prices)

    # Strategy params
    entry_z = 2.0
    exit_z  = 0.5
    q = 1e-3
    r = 1e-3
    sizing = 0.40            # 40% per leg
    costs_bps = 12.5         # 0.125%
    borrow_annual = 0.0025   # 0.25% p.a.

    def run_segment(name, seg):
        x = seg.iloc[:, 0]
        y = seg.iloc[:, 1]
        print(f"\n📊 {name} summary:")
        metrics = run_backtest(
            x, y,
            costs_bps=12.5,
            borrow_annual=borrow_annual

        )
        print(metrics)
        return x, y, metrics

    # -----------------------------
    # 4) Walk-forward run
    # -----------------------------
    _, _, train_res = run_segment("TRAIN (60%)", train)
    _, _, test_res  = run_segment("TEST  (20%)",  test)
    x_valid, y_valid, valid_res = run_segment("VALID (20%)", valid)

    # -----------------------------
    # 5) Final plot on Valid only
    # -----------------------------
    plot_results(
        x_valid,
        y_valid,
        valid_res,
        entry_z,
        exit_z
    )
    print("\n✅ DONE. Walk-forward finished.")

if __name__ == "__main__":
    main()
