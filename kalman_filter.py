import numpy as np
from pykalman import KalmanFilter


def run_kalman(x, y, q=0.001, r=0.001):
    """
    Returns:
        dict with keys: beta, alpha, spread, zscore
    """

    # Convert to numpy
    x = np.array(x).flatten()
    y = np.array(y).flatten()

    # Drop NaNs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    # Kalman Filter setup
    # State: [beta, alpha]
    transition_matrix = np.eye(2)  # Keep prev state
    transition_covariance = q * np.eye(2)

    # Obs model: x_t = beta * y_t + alpha
    observation_matrix = np.column_stack([y, np.ones_like(y)])
    observation_covariance = r * np.eye(1)

    # Reshape obs_matrix for pykalman: (timesteps, 1, 2)
    observation_matrix = observation_matrix.reshape((-1, 1, 2))

    # Initialize Kalman Filter
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.eye(2)
    )

    # Run filter
    state_means, _ = kf.smooth(x)

    beta = state_means[:, 0]
    alpha = state_means[:, 1]

    # Spread
    spread = x - (beta * y + alpha)

    # Z-score
    spread_mean = np.nanmean(spread)
    spread_std = np.nanstd(spread)
    zscore = (spread - spread_mean) / spread_std

    return {
        "beta": beta,
        "alpha": alpha,
        "spread": spread,
        "zscore": zscore,
        "clean_x": x,
        "clean_y": y
    }
