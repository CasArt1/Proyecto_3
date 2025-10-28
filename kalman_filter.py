# ==========================================================
# kalman_filter.py
# ==========================================================
# Implements a two-state Kalman Filter to dynamically
# estimate intercept (beta0) and hedge ratio (beta1)
# between two assets.
# ==========================================================

import numpy as np
import pandas as pd
import os

class KalmanFilterHedgeRatio:
    def __init__(self, q=0.001, r=0.001, save_path=None):
        """
        q : process noise (controls beta drift speed)
        r : measurement noise (controls sensitivity to errors)
        save_path : optional CSV file to store beta time series
        """
        # Initial state [intercept, hedge_ratio]
        self.x = np.array([0.0, 1.0])
        # State covariance matrix
        self.P = np.eye(2) * 100.0
        # Process noise covariance
        self.Q = np.eye(2) * q
        # Measurement noise covariance
        self.R = np.array([[r]])
        # State transition matrix
        self.A = np.eye(2)
        # Measurement matrix (updated every observation)
        self.C = None

        # Keep record of evolution
        self.history = []
        self.save_path = save_path

    def predict(self):
        """Predict step"""
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x_obs, y_obs):
        """Update step given new observation (x_obs, y_obs)"""
        self.C = np.array([[1.0, x_obs]])
        y_pred = self.C @ self.x
        innovation = y_obs - y_pred
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(2) - K @ self.C) @ self.P

        # Save results
        self.history.append((self.x[0], self.x[1]))  # intercept, beta
        return self.x[1]  # current hedge ratio

    def get_hedge_series(self):
        """Return DataFrame of intercept and hedge ratio over time"""
        df = pd.DataFrame(self.history, columns=["intercept", "beta"])
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            df.to_csv(self.save_path, index=False)
            print(f"💾 Saved Kalman beta evolution to {self.save_path}")
        return df
