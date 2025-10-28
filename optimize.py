# ==========================================================
# optimize.py
# ==========================================================
# Uses Optuna to tune Kalman filter noise parameters (Q, R)
# and trading thresholds (entry_z, exit_z) to maximize Sharpe.
# ==========================================================

import optuna
from backtest import run_backtest
from kalman_filter import KalmanFilterHedgeRatio

def optimize_parameters(stock_x, stock_y, n_trials=50):
    """
    Uses Optuna to find the best (q, r, entry_z, exit_z) combination
    that maximizes the Sharpe ratio from the backtest.
    """

    def objective(trial):
        # Sample parameters
        q = trial.suggest_loguniform("q", 1e-6, 1e-2)
        r = trial.suggest_loguniform("r", 1e-4, 1e-1)
        entry_z = trial.suggest_uniform("entry_z", 1.0, 3.0)
        exit_z = trial.suggest_uniform("exit_z", 0.1, 1.0)

        try:
            result = run_backtest(
                stock_x, stock_y,
                q=q, r=r,
                entry_z=entry_z, exit_z=exit_z,
                kalman_cls=KalmanFilterHedgeRatio
            )
            sharpe = result["Sharpe"]
        except Exception:
            sharpe = -999  # penalize failed trials

        return sharpe

    print("🚀 Starting Optuna optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print("\n🏆 Best parameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.6f}")
    print(f"📈 Best Sharpe ratio: {study.best_value:.4f}")

    return best_params, study.best_value
