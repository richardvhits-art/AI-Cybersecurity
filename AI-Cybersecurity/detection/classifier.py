"""
Module 5: Threat Classification
----------------------------------
Random Forest classifier for threat category, used ONLY when the supplied
dataset has labeled data. If no labels exist, classification is skipped
and the system clearly reports that only anomaly detection is available.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split


class ThreatClassifier:
    """Wraps sklearn's RandomForestClassifier with train/predict/save/load + metrics."""

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",  # important: cybersecurity datasets are usually imbalanced
        )
        self.classes_: list[str] = []
        self._is_fitted = False

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.25,
              random_state: int = 42) -> dict:
        """
        Train/test split, fit, and return evaluation metrics.
        Uses macro-averaged precision/recall/F1 since classes are imbalanced --
        raw accuracy alone would be misleading (see README: class imbalance).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        self.classes_ = list(self.model.classes_)

        y_pred = self.model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=self.classes_).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, labels=self.classes_, zero_division=0
            ),
            "classes": self.classes_,
        }
        return metrics

    def predict(self, X: np.ndarray) -> list[dict]:
        """Returns list of {"threat_category": ..., "confidence": float}."""
        if not self._is_fitted:
            raise RuntimeError("ThreatClassifier.predict() called before training/loading.")

        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)

        results = []
        for pred, proba_row in zip(predictions, probabilities):
            confidence = float(np.max(proba_row))
            results.append({
                "threat_category": str(pred),
                "confidence": round(confidence, 4),
            })
        return results

    def save(self, path: str = "models/random_forest.joblib") -> str:
        if not self._is_fitted:
            raise RuntimeError("Cannot save an untrained classifier.")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str = "models/random_forest.joblib") -> "ThreatClassifier":
        model = joblib.load(path)
        if not isinstance(model, ThreatClassifier):
            raise TypeError("Loaded object is not a ThreatClassifier instance.")
        return model
