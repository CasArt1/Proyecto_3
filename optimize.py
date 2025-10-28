import json
import optuna
import numpy as np
import pandas as pd
from data_loader import download_data as load_data
from backtest import run_backtest
from pair_selection import find_top_pairs

# ✅ Config
COSTS_BPS = 12.5
BORROW_ANNUAL = 0.25
TOP_K = 5

# ✅ Objective for Optuna (Sharpe Ratio maximization)
def objective(trial, x, y):
    # Entry Z: allow slightly wide bands
    z_entry = trial.suggest_float("z_entry", 1.5, 2.2)
    z_exit  = trial.suggest_float("z_exit", 0.3, 0.8)
    log10_q = trial.suggest_float("log10_q", -4, -1)
    log10_r = trial.suggest_float("log10_r", -6, -2)
    # R ~ 1e-6 to 1e-3


    q = 10 ** log10_q
    r = 10 ** log10_r

    try:
        result = run_backtest(
            x, y,
            q=q, r=r,
            z_entry=z_entry,
            z_exit=z_exit,
            costs_bps=COSTS_BPS,
            borrow_annual=BORROW_ANNUAL
        )
        sharpe = float(result.get("sharpe_daily", 0))
        return -abs(sharpe)  # minimize negative Sharpe
    except Exception as e:
        print("⚠️ Error:", e)
        return 9999

# ✅ Main optimization routine
def optimize_pair(x_ticker, y_ticker, closes):
    print(f"\n🚀 Optimizing {x_ticker}-{y_ticker} ...")
    x, y = closes[x_ticker].dropna(), closes[y_ticker].dropna()

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, x, y), n_trials=100, show_progress_bar=False)

    best = study.best_params
    best["best_sharpe"] = -study.best_value
    print(f"🥇 Best Sharpe for {x_ticker}-{y_ticker}: {best['best_sharpe']:.4f}")
    print(f"Best params: {json.dumps(best, indent=2)}")
    return best

# ✅ Runner for Top-5 pairs
def main():
    closes = load_data()
    top_pairs = find_top_pairs(closes, top_n=TOP_K)
    all_results = {}

    for _, row in top_pairs.iterrows():
        x_t = row["x"]
        y_t = row["y"]
        best = optimize_pair(x_t, y_t, closes)
        all_results[f"{x_t}-{y_t}"] = best

    os.makedirs("data", exist_ok=True)
    with open("data/best_kf_params_top5.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n💾 Saved all Top-5 best parameter sets → data/best_kf_params_top5.json")

if __name__ == "__main__":
    import os
    main()
