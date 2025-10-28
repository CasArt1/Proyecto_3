# kalman_filter.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def run_kalman(price_x: pd.Series,
               price_y: pd.Series,
               q: float = 1e-3,
               r: float = 1e-3,
               lookback_beta: int = 63,
               lookback_z: int = 63) -> dict:
    """
    Rolling OLS to approximate dynamic hedge ratio.
    Returns dict with pandas Series:
      - 'hedge_ratio'  : rolling beta(t)
      - 'spread'       : x - beta*y (using beta(t))
      - 'zscore'       : z of spread with rolling mean/std
    """
    df = pd.concat([price_x.rename("x"), price_y.rename("y")], axis=1).dropna()
    x = np.log(df["x"].values)
    y = np.log(df["y"].values)

    betas = np.full(len(df), np.nan, dtype=float)
    for i in range(lookback_beta, len(df)):
        Xw = y[i - lookback_beta:i].reshape(-1, 1)
        yw = x[i - lookback_beta:i]
        try:
            model = LinearRegression().fit(Xw, yw)
            betas[i] = float(model.coef_[0])
        except Exception:
            betas[i] = np.nan

    # forward-fill a bit to avoid gaps at trade time
    betas = pd.Series(betas, index=df.index).ffill()

    spread = pd.Series(x, index=df.index) - betas * pd.Series(y, index=df.index)

    m = spread.rolling(lookback_z).mean()
    s = spread.rolling(lookback_z).std()
    z = (spread - m) / s

    return {
        "hedge_ratio": betas,
        "spread": spread,
        "zscore": z
    }
