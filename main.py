# main.py
import json
from pathlib import Path

from data_loader import load_pair_prices, split_data
from backtest import run_backtest
from visualize import plot_results


def run_segment(name, x, y, q, r, z_entry, z_exit):
    print(f"\n📊 {name} summary:")
    metrics = run_backtest(
        x, y,
        q=q, r=r,
        entry_z=z_entry,
        exit_z=z_exit,
        sizing=0.40,
        costs_bps=12.5,
        borrow_annual=0.0025
    )
    print(metrics)
    return metrics


def main():
    # Load pair prices (already saved as data/stocks.csv by pair selection)
    closes = load_pair_prices("data/stocks.csv")
    train, test, valid = split_data(closes)

    x_train, y_train = train.iloc[:, 0], train.iloc[:, 1]
    x_test, y_test = test.iloc[:, 0], test.iloc[:, 1]
    x_valid, y_valid = valid.iloc[:, 0], valid.iloc[:, 1]

    # Load optimized KF + Z band parameters
    params_path = Path("data/best_kf_params.json")
    if not params_path.exists():
        raise FileNotFoundError("⚠️ best_kf_params.json not found! Run optimize.py first.")

    params = json.loads(params_path.read_text())
    q = 10.0 ** params["log10_q"]
    r = 10.0 ** params["log10_r"]
    z_entry = params["z_entry"]
    z_exit = params["z_exit"]

    print("\n✅ Using Optimized Parameters:")
    print(f"Q = {q:.2e}, R = {r:.2e}, Entry Z = {z_entry:.2f}, Exit Z = {z_exit:.2f}")

    # Optionally see TRAIN results using tuned params
    run_segment("TRAIN (60%)", x_train, y_train, q, r, z_entry, z_exit)

    # PERFORMANCE ON TEST + VALIDATION
    test_res = run_segment("TEST (20%)", x_test, y_test, q, r, z_entry, z_exit)
    valid_res = run_segment("VALID (20%)", x_valid, y_valid, q, r, z_entry, z_exit)

    # ✅ Plot the strategy using TEST set only
    plot_results(x_test, y_test, test_res, z_entry, z_exit)

    print("\n✅ SUCCESS! Review TEST and VALID results above.")
    print("📈 Chart window shown for TEST segment. Close it to end.")


if __name__ == "__main__":
    main()

