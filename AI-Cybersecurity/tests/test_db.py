import os

import pytest

from database import db


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db, "DB_PATH", db_path)
    db.init_db()
    return db_path


def test_insert_and_get_alert(temp_db):
    alert_id = db.insert_alert({
        "timestamp": "2024-01-01T00:00:00",
        "source_ip": "1.2.3.4",
        "destination_ip": "5.6.7.8",
        "destination_port": 443,
        "anomaly_score": 0.8,
        "prediction": "anomaly",
        "threat_category": "port_scan",
        "confidence": 0.9,
        "risk_score": 75,
        "severity": "HIGH",
    })
    alert = db.get_alert(alert_id)
    assert alert is not None
    assert alert["severity"] == "HIGH"


def test_list_alerts_filters_by_severity(temp_db):
    db.insert_alert({"severity": "LOW", "prediction": "normal", "status": "open"})
    db.insert_alert({"severity": "HIGH", "prediction": "anomaly", "status": "open"})

    high_alerts = db.list_alerts(severity="HIGH")
    assert len(high_alerts) == 1
    assert high_alerts[0]["severity"] == "HIGH"


def test_insert_and_get_ai_explanation(temp_db):
    alert_id = db.insert_alert({"severity": "MEDIUM", "prediction": "anomaly"})
    db.insert_ai_explanation(alert_id, {
        "summary": "Test summary",
        "why_suspicious": ["reason 1", "reason 2"],
        "possible_category": "unknown",
        "severity_explanation": "explanation",
        "recommended_investigation": ["step 1"],
    })
    explanation = db.get_ai_explanation(alert_id)
    assert explanation["summary"] == "Test summary"
    assert explanation["why_suspicious"] == ["reason 1", "reason 2"]


def test_get_statistics_counts(temp_db):
    db.insert_alert({"severity": "LOW", "prediction": "normal"})
    db.insert_alert({"severity": "CRITICAL", "prediction": "anomaly"})
    stats = db.get_statistics()
    assert stats["total_events"] == 2
    assert stats["total_anomalies"] == 1
