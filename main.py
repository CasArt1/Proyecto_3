# main.py — Clean global tech workflow

from data_loader import load_pair_prices
from pair_selection import find_best_pair
from backtest import run_backtest_splits
from visualize import plot_results

def main():
    print("🔍 Selecting best global tech pair…")
    best_pair = find_best_pair()  # ✅ No tickers argument
    x, y = best_pair

    print("\n📥 Loading selected pair price data…")
    df = load_pair_prices()  # ✅ Reads data/stocks.csv from selection step

    # Split and backtest
    train_res, test_res, valid_res = run_backtest_splits(df, x, y)

    print("\n📊 Results:")
    print("TRAIN:", train_res)
    print("TEST:", test_res)
    print("VALID:", valid_res)

    print("\n📈 Plotting results...")
    plot_results(df, x, y, train_res, test_res, valid_res)

    print("\n✅ Done!")

if __name__ == "__main__":
    main()
