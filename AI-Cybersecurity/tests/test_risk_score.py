from detection.risk_score import calculate_risk_score, classify_severity


def test_normal_event_scores_low():
    result = calculate_risk_score(
        is_anomaly=False, anomaly_score=0.1, failed_logins=0,
        unique_dest_ports=1, connection_frequency=1, total_bytes=1000,
    )
    assert result["severity"] in ("LOW", "MEDIUM")
    assert result["score_label"] == "Application Risk Score"


def test_high_anomaly_scores_higher():
    low = calculate_risk_score(is_anomaly=False, anomaly_score=0.0)
    high = calculate_risk_score(
        is_anomaly=True, anomaly_score=0.9, failed_logins=25,
        unique_dest_ports=40, connection_frequency=50, total_bytes=200000,
    )
    assert high["risk_score"] > low["risk_score"]


def test_severity_thresholds():
    assert classify_severity(10) == "LOW"
    assert classify_severity(45) == "MEDIUM"
    assert classify_severity(70) == "HIGH"
    assert classify_severity(95) == "CRITICAL"


def test_score_is_bounded_0_to_100():
    result = calculate_risk_score(
        is_anomaly=True, anomaly_score=1.0, failed_logins=1000,
        unique_dest_ports=1000, connection_frequency=1000,
        total_bytes=10_000_000, classifier_confidence=1.0,
    )
    assert 0 <= result["risk_score"] <= 100
