import json
import pandas as pd
from data_loader import download_data
from backtest import run_backtest
from visualize import plot_results

TOP5_PARAMS_FILE = "data/best_kf_params_top5.json"
TOP_PAIRS_FILE = "data/top_pairs.csv"

def main():
    # Load prices
    closes = download_data()

    # Load optimized params
    with open(TOP5_PARAMS_FILE, "r") as f:
        best_params_dict = json.load(f)

    # Load Top-5 pairs
    top_pairs = pd.read_csv(TOP_PAIRS_FILE)

    results = []

    # Loop through Top-5 pairs
    for _, row in top_pairs.iterrows():
        x_t, y_t = row["x"], row["y"]
        print(f"\n📌 Evaluating pair: {x_t}-{y_t}")

        key = f"{x_t}-{y_t}"
        params = best_params_dict.get(key)
        if not params:
            print(f"⚠️ No params for {x_t}-{y_t}, skipping.")
            continue

        Q = 10 ** params["log10_q"]
        R = 10 ** params["log10_r"]
        entry_z = params["z_entry"]
        exit_z = params["z_exit"]

        # Clean/aligned series
        df = closes[[x_t, y_t]].dropna()
        x, y = df[x_t], df[y_t]

        n = len(df)
        train_split = int(n * 0.6)
        test_split = int(n * 0.8)

        x_train, x_test, x_val = x[:train_split], x[train_split:test_split], x[test_split:]
        y_train, y_test, y_val = y[:train_split], y[train_split:test_split], y[test_split:]

        metrics_valid = run_backtest(x_val, y_val, Q, R, entry_z, exit_z)

        # Filter out broken results
        if (metrics_valid is None or
            pd.isna(metrics_valid.get("sharpe_daily", None)) or
            metrics_valid.get("trades", 0) == 0):
            print(f"⚠️ {x_t}-{y_t} invalid, skipping.")
            continue

        # ✅ Store everything for plotting later!
        results.append({
            "pair": f"{x_t}-{y_t}",
            "entry_z": entry_z,
            "exit_z": exit_z,
            "x_test": x_test,
            "y_test": y_test,
            **metrics_valid
        })

        print(f"✅ {x_t}-{y_t} VALID results:")
        print(metrics_valid)

    print("\n🏁 FINAL VALIDATION COMPLETE ✅")
    print("\n🔥 Ranking pairs by Sharpe:")

    if len(results) == 0:
        print("❌ No valid pairs survived. Exiting.")
        return

    # Sort
    results = sorted(results, key=lambda r: r["sharpe_daily"], reverse=True)

    for r in results:
        print(f"{r['pair']:>10} | Sharpe: {r['sharpe_daily']:.4f} | Return: {r['total_return_pct']:.2f}%")

    # ✅ Plot best pair
    best = results[0]
    pair = best["pair"]

    print(f"\n📈 Plotting BEST pair: {pair}")

    plot_results(
        best,
        best["x_test"],
        best["y_test"],
        f"BEST VALID — {pair}",
        best["entry_z"],
        best["exit_z"]
    )


if __name__ == "__main__":
    main()
