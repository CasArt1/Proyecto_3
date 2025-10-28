# optimize.py
import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from data_loader import load_pair_prices, split_data
from backtest import run_backtest


def load_train_segment(csv_path: str = "data/stocks.csv"):
    """Load pair prices from CSV and return (x, y) for the TRAIN (60%) split."""
    closes = load_pair_prices(csv_path)  # Date index, 2 columns
    train, _, _ = split_data(closes)
    x = train.iloc[:, 0]
    y = train.iloc[:, 1]
    return x, y


def objective(trial: optuna.Trial) -> float:
    """
    Maximize Sharpe on TRAIN set by tuning:
      - Kalman Filter noises: Q, R
      - Trading bands: z_entry, z_exit
    """
    # --- Search space ---
    # log-uniform for Q, R (process/measurement noise)
    log10_q = trial.suggest_float("log10_q", -8.0, -2.0)
    log10_r = trial.suggest_float("log10_r", -8.0, -2.0)
    q = 10.0 ** log10_q
    r = 10.0 ** log10_r

    # Trading bands (ensure exit < entry)
    z_entry = trial.suggest_float("z_entry", 1.2, 3.0)
    z_exit_hi = min(1.2, z_entry - 0.1)  # keep a gap to avoid churning
    z_exit = trial.suggest_float("z_exit", 0.1, z_exit_hi)

    # --- Run backtest on TRAIN set only ---
    x, y = load_train_segment()
    metrics = run_backtest(
        x, y,
        q=q, r=r,
        entry_z=z_entry,
        exit_z=z_exit,
        sizing=0.40,           # 40% per leg (your choice)
        costs_bps=12.5,        # 0.125%
        borrow_annual=0.0025   # 0.25% p.a.
    )

    sharpe = float(metrics.get("sharpe_daily", np.nan))
    # Defensive: If backtest blew up / NaN, penalize
    if not np.isfinite(sharpe):
        return -1e9

    # You can add a soft penalty on insane drawdowns:
    mdd = float(metrics.get("max_drawdown_pct", 0.0))
    # (Optional) penalize if MDD < -80%
    if mdd < -0.80:
        sharpe += mdd  # lower objective a bit

    return sharpe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Make sure the pair CSV exists
    if not Path("data/stocks.csv").exists():
        raise FileNotFoundError(
            "data/stocks.csv not found. Run main.py first to select & save the best pair."
        )

    # Optuna study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("\n🥇 Best Sharpe:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k} = {v}")

    # Save best params
    best_params_path = Path("data/best_kf_params.json")
    best_params_path.write_text(json.dumps(study.best_params, indent=2))
    print(f"\n💾 Saved best params → {best_params_path}")

    # Re-run TRAIN backtest with best params, then show TEST/VALID recommendation
    x, y = load_train_segment()
    p = study.best_params
    q = 10.0 ** p["log10_q"]
    r = 10.0 ** p["log10_r"]
    z_entry = p["z_entry"]
    z_exit = p["z_exit"]

    metrics = run_backtest(
        x, y,
        q=q, r=r,
        entry_z=z_entry,
        exit_z=z_exit,
        sizing=0.40,
        costs_bps=12.5,
        borrow_annual=0.0025
    )
    print("\n🔁 TRAIN metrics with best params:", metrics)
    print("\nNext step: Plug the best params into main.py and re-run TEST + VALIDATION.")


if __name__ == "__main__":
    main()
