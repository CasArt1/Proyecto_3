import numpy as np

class KalmanFilterSDA:
    """
    Implementación del Filtro de Kalman para la estimación dinámica del hedge ratiom (beta_1).

    El estado (x) es el hedge ratio beta_1, y opcionalmente el intercepto beta_o.
    """

    def __init__(self, initial_beta: float, initial_R: float, Q_process: float, R_measurement: float):
        """
        Inicializa el Filtro de Kalman.

        Args:
            initial_beta (float): El coeficiente inicial de OLS (beta_1, de la Fase 1).
            initial_R (float): La varianza inicial del spread (residuales de OLS).
            Q_process (float): Varianza del ruido del proceso (Q).
            R_measurement (float): Varianza del ruido de medición (R).
        """

        self.x = np.array([initial_beta])
        self.P = np.array([[initial_R]])
        self.F = np.array([[1.0]])
        self.Q = np.array([[Q_process]])
        self.R = np.array([[R_measurement]])

    def predict(self) -> tuple[float, float]:
        """
        Paso de predicción del Filtro de Kalman. Estima el estado y su covarianza en t+1.

        Equations:
            x_pred = F *x_t
            P_pred = F * P_t * F^T + Q
        Returns:
            tuple[float, float]: Predicción del estado (beta_1) y su varianza.
        """
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        self.x = x_pred
        self.P = P_pred
        return self.x[0], self.P[0, 0] # Devuelve beta_1 y su varianza predicha
    
    def update(self, price_y: float, price_x: float) -> tuple[float, float, float]:
        """
        Paso de actualización del Filtro de Kalman. Ajusta la predicción con la nueva observación.

        Args:
            price_y (float): Precio del activo Y en t+1 (activo 1, dependiente, P1).
            price_x (float): Precio del activo X en t+1 (activo 2, independiente, P2).

        Equations:
            Innovation (y_tilde) = y_t - H * x_pred
            S = H * P_pred * H^T + R
            Kalman Gain (K) = P_pred * H^T * S^-1
            x_update = x_pred + K * y_tilde
            P_update = (I - K * H) * P_pred
        """

        H = np.array([[price_x]])  # Observación del modelo
        spread_pred = H @ self.x  # Predicción del spread
        y_tilde = price_y - spread_pred  # Innovación

        S = H @ self.P @ H.T + self.R  # Covarianza de la innovación
        K = self.P @ H.T * (1.0 / S)  # Ganancia de Kalman

        self.x = self.x + K * y_tilde  # Actualización del estado
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ H) @ self.P  # Actualización de la covarianza

        return self.x[0], self.P[0, 0], S[0, 0]
    
    def get_hedge_ratio(self) -> float:
        """
        Returns the current hedge ratio (beta_1).
        
        Returns:
            float: Current hedge ratio estimate.
        """
        return self.x[0]
