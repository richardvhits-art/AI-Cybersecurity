"""
Module 4: Machine Learning - Anomaly Detection
-------------------------------------------------
Isolation Forest based anomaly detector. This is the PRIMARY detection
mechanism in the system. Gemini (Module 7) never replaces this -- it only
explains results this module produces.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Wraps sklearn's IsolationForest with train/predict/save/load convenience methods."""

    def __init__(self, n_estimators: int = 200, contamination: float = 0.1,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._is_fitted = False

    def train(self, X: np.ndarray) -> "AnomalyDetector":
        """Fit the Isolation Forest on preprocessed feature matrix X."""
        self.model.fit(X)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> list[dict]:
        """
        Predict anomaly status for each row in X.
        Returns a list of {"prediction": "normal"|"anomaly", "anomaly_score": float}.
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyDetector.predict() called before training/loading a model.")

        raw_predictions = self.model.predict(X)          # 1 = normal, -1 = anomaly
        raw_scores = self.model.decision_function(X)      # higher = more normal

        results = []
        for pred, score in zip(raw_predictions, raw_scores):
            # Normalize so a HIGHER anomaly_score means MORE anomalous (0-1 range, approx)
            normalized_score = float(np.clip(0.5 - score, 0, 1))
            results.append({
                "prediction": "anomaly" if pred == -1 else "normal",
                "anomaly_score": round(normalized_score, 4),
            })
        return results

    def save(self, path: str = "models/isolation_forest.joblib") -> str:
        if not self._is_fitted:
            raise RuntimeError("Cannot save an untrained model.")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str = "models/isolation_forest.joblib") -> "AnomalyDetector":
        model = joblib.load(path)
        if not isinstance(model, AnomalyDetector):
            raise TypeError("Loaded object is not an AnomalyDetector instance.")
        return model
