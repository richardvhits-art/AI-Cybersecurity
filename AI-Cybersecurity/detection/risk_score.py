"""
Module 6: Risk Scoring
-------------------------
Deterministic 0-100 "Application Risk Score" (NOT an industry-standard score
like CVSS -- explicitly labeled as such). All weights are configurable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskWeights:
    """Configurable weights. Values are illustrative starting points, not
    validated against real-world incident data -- tune for your environment."""
    anomaly_base: float = 30.0          # points if flagged as anomaly at all
    anomaly_magnitude: float = 25.0     # scaled by anomaly_score (0-1)
    failed_logins: float = 15.0         # scaled by normalized failed-login activity
    unique_ports: float = 10.0          # scaled by normalized port-scan-like behavior
    connection_frequency: float = 10.0  # scaled by normalized connection frequency
    traffic_volume: float = 5.0         # scaled by normalized byte volume
    classifier_confidence: float = 5.0  # scaled by classifier confidence, if available


def _normalize(value: float, soft_max: float) -> float:
    """Clamp value/soft_max into [0, 1]. soft_max is a practical ceiling, not a hard limit."""
    if soft_max <= 0:
        return 0.0
    return max(0.0, min(1.0, value / soft_max))


def calculate_risk_score(
    is_anomaly: bool,
    anomaly_score: float,
    failed_logins: float = 0.0,
    unique_dest_ports: float = 0.0,
    connection_frequency: float = 0.0,
    total_bytes: float = 0.0,
    classifier_confidence: float | None = None,
    weights: RiskWeights | None = None,
) -> dict:
    """
    Compute a deterministic Application Risk Score (0-100).
    Inputs are raw evidence values; this function normalizes and weights them.
    """
    if weights is None:
        weights = RiskWeights()

    score = 0.0

    if is_anomaly:
        score += weights.anomaly_base
        score += weights.anomaly_magnitude * max(0.0, min(1.0, anomaly_score))

    score += weights.failed_logins * _normalize(failed_logins, soft_max=20)
    score += weights.unique_ports * _normalize(unique_dest_ports, soft_max=30)
    score += weights.connection_frequency * _normalize(connection_frequency, soft_max=50)
    score += weights.traffic_volume * _normalize(total_bytes, soft_max=200_000)

    if classifier_confidence is not None:
        score += weights.classifier_confidence * max(0.0, min(1.0, classifier_confidence))

    final_score = int(round(max(0.0, min(100.0, score))))
    severity = classify_severity(final_score)

    return {
        "risk_score": final_score,
        "severity": severity,
        "score_label": "Application Risk Score",  # explicitly NOT an industry-standard score
    }


def classify_severity(risk_score: int) -> str:
    if risk_score <= 30:
        return "LOW"
    if risk_score <= 60:
        return "MEDIUM"
    if risk_score <= 80:
        return "HIGH"
    return "CRITICAL"
