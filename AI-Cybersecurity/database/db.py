"""
Module 8: Database
----------------------
SQLite persistence layer. All queries use parameterized statements --
never raw string concatenation -- to prevent SQL injection.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.getenv("DATABASE_URL", "database/cyberdetect.db").replace("sqlite:///", "")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL DEFAULT 'analyst',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS network_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    source_ip TEXT,
    destination_ip TEXT,
    destination_port INTEGER,
    raw_data TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    source_ip TEXT,
    destination_ip TEXT,
    destination_port INTEGER,
    anomaly_score REAL,
    prediction TEXT,
    threat_category TEXT,
    confidence REAL,
    risk_score INTEGER,
    severity TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    output TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (alert_id) REFERENCES alerts (id)
);

CREATE TABLE IF NOT EXISTS ai_explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER NOT NULL,
    summary TEXT,
    why_suspicious TEXT,
    possible_category TEXT,
    severity_explanation TEXT,
    recommended_investigation TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (alert_id) REFERENCES alerts (id)
);
"""


@contextmanager
def get_connection():
    """Context-managed SQLite connection. Ensures commit/close even on error."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't already exist. Safe to call on every startup."""
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_alert(alert: dict) -> int:
    """Insert an alert row. Returns the new alert id."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO alerts (
                timestamp, source_ip, destination_ip, destination_port,
                anomaly_score, prediction, threat_category, confidence,
                risk_score, severity, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert.get("timestamp"),
                alert.get("source_ip"),
                alert.get("destination_ip"),
                alert.get("destination_port"),
                alert.get("anomaly_score"),
                alert.get("prediction"),
                alert.get("threat_category"),
                alert.get("confidence"),
                alert.get("risk_score"),
                alert.get("severity"),
                alert.get("status", "open"),
                _now(),
            ),
        )
        return cursor.lastrowid


def get_alert(alert_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return dict(row) if row else None


def list_alerts(severity: str | None = None, threat_category: str | None = None,
                 status: str | None = None, limit: int = 200) -> list[dict]:
    """Filtered alert listing. All filters use parameterized queries."""
    query = "SELECT * FROM alerts WHERE 1=1"
    params: list = []

    if severity:
        query += " AND severity = ?"
        params.append(severity)
    if threat_category:
        query += " AND threat_category = ?"
        params.append(threat_category)
    if status:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def insert_ai_explanation(alert_id: int, explanation: dict) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO ai_explanations (
                alert_id, summary, why_suspicious, possible_category,
                severity_explanation, recommended_investigation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert_id,
                explanation.get("summary"),
                json.dumps(explanation.get("why_suspicious", [])),
                explanation.get("possible_category"),
                explanation.get("severity_explanation"),
                json.dumps(explanation.get("recommended_investigation", [])),
                _now(),
            ),
        )
        return cursor.lastrowid


def get_ai_explanation(alert_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM ai_explanations WHERE alert_id = ? ORDER BY created_at DESC LIMIT 1",
            (alert_id,),
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["why_suspicious"] = json.loads(result["why_suspicious"] or "[]")
        result["recommended_investigation"] = json.loads(result["recommended_investigation"] or "[]")
        return result


def get_statistics() -> dict:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM alerts").fetchone()["c"]
        anomalies = conn.execute(
            "SELECT COUNT(*) AS c FROM alerts WHERE prediction = 'anomaly'"
        ).fetchone()["c"]

        severity_counts = {}
        for row in conn.execute("SELECT severity, COUNT(*) AS c FROM alerts GROUP BY severity"):
            severity_counts[row["severity"]] = row["c"]

        category_counts = {}
        for row in conn.execute(
            "SELECT threat_category, COUNT(*) AS c FROM alerts "
            "WHERE threat_category IS NOT NULL GROUP BY threat_category"
        ):
            category_counts[row["threat_category"]] = row["c"]

        return {
            "total_events": total,
            "total_anomalies": anomalies,
            "severity_counts": severity_counts,
            "threat_category_counts": category_counts,
        }
