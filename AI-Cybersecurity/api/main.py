"""
Module 9: FastAPI Backend
----------------------------
Exposes the ML pipeline + Gemini explanation layer over HTTP.
Gemini is called ONLY on explicit request (/alerts/{id}/explain),
never automatically for every event.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from ai.gemini_client import GeminiUnavailableError, generate_security_explanation
from database import db
from detection.anomaly_detector import AnomalyDetector
from detection.classifier import ThreatClassifier
from detection.risk_score import calculate_risk_score
from preprocessing.clean import load_preprocessor, transform_with_existing
from preprocessing.features import engineer_features
from preprocessing.ingest import DataIngestionError, load_dataset, validate_dataset

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api.main")

app = FastAPI(
    title="AI-Powered Cybersecurity Threat Detection & Alert Explanation System",
    description="Defensive security monitoring: ML anomaly detection + Gemini-based alert explanation.",
    version="1.0.0",
)

MODELS_DIR = Path("models")
UPLOAD_DIR = Path("data/raw")

_anomaly_model: Optional[AnomalyDetector] = None
_classifier_model: Optional[ThreatClassifier] = None
_preprocessor = None
_numeric_cols: list[str] = []
_categorical_cols: list[str] = []


# ---------- Pydantic models ----------

class NetworkEvent(BaseModel):
    """A single event to analyze. Extra fields are ignored, not rejected,
    so datasets with slightly different schemas still work."""
    timestamp: Optional[str] = None
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    destination_port: Optional[int] = None
    protocol: Optional[str] = None
    bytes_sent: Optional[float] = Field(default=None, ge=0)
    bytes_received: Optional[float] = Field(default=None, ge=0)
    packet_count: Optional[float] = Field(default=None, ge=0)
    flow_duration: Optional[float] = Field(default=None, ge=0)
    failed_logins: Optional[float] = Field(default=None, ge=0)

    class Config:
        extra = "allow"


class AnalyzeResponse(BaseModel):
    alert_id: int
    prediction: str
    anomaly_score: float
    threat_category: Optional[str] = None
    confidence: Optional[float] = None
    risk_score: int
    severity: str


# ---------- Startup ----------

@app.on_event("startup")
def startup() -> None:
    db.init_db()
    _load_models_if_available()


def _load_models_if_available() -> None:
    """Load trained models at startup, if present. Does NOT train -- see train.py."""
    global _anomaly_model, _classifier_model, _preprocessor, _numeric_cols, _categorical_cols

    try:
        _anomaly_model = AnomalyDetector.load(str(MODELS_DIR / "isolation_forest.joblib"))
        logger.info("Loaded Isolation Forest model.")
    except FileNotFoundError:
        logger.warning("No Isolation Forest model found. Run `python train.py` first.")

    try:
        _classifier_model = ThreatClassifier.load(str(MODELS_DIR / "random_forest.joblib"))
        logger.info("Loaded Random Forest classifier.")
    except FileNotFoundError:
        logger.info("No classifier model found -- classification will be unavailable.")

    try:
        _preprocessor, _numeric_cols, _categorical_cols = load_preprocessor(
            str(MODELS_DIR / "preprocessor.joblib")
        )
        logger.info("Loaded preprocessing pipeline.")
    except FileNotFoundError:
        logger.warning("No preprocessor found. Run `python train.py` first.")


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "anomaly_model_loaded": _anomaly_model is not None,
        "classifier_loaded": _classifier_model is not None,
        "preprocessor_loaded": _preprocessor is not None,
    }


@app.get("/models/status")
def models_status():
    return {
        "isolation_forest": "loaded" if _anomaly_model else "missing - run python train.py",
        "random_forest": "loaded" if _classifier_model else "unavailable (optional)",
        "preprocessor": "loaded" if _preprocessor else "missing - run python train.py",
    }


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".csv", ".json")):
        raise HTTPException(400, "Only .csv and .json files are supported.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        df = load_dataset(str(dest))
        report = validate_dataset(df)
    except DataIngestionError as e:
        raise HTTPException(400, str(e))

    return {"filename": file.filename, "validation_report": report}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_event(event: NetworkEvent):
    if _anomaly_model is None or _preprocessor is None:
        raise HTTPException(
            503, "ML model not trained. Run `python train.py` first."
        )

    import pandas as pd
    df = pd.DataFrame([event.dict()])
    df, _added_features = engineer_features(df)

    try:
        X = transform_with_existing(df, _preprocessor, _numeric_cols, _categorical_cols)
    except Exception as e:
        raise HTTPException(400, f"Feature preprocessing failed: {e}")

    anomaly_result = _anomaly_model.predict(X)[0]

    threat_category, confidence = None, None
    if _classifier_model is not None:
        clf_result = _classifier_model.predict(X)[0]
        threat_category = clf_result["threat_category"]
        confidence = clf_result["confidence"]

    total_bytes = (event.bytes_sent or 0) + (event.bytes_received or 0)
    risk = calculate_risk_score(
        is_anomaly=(anomaly_result["prediction"] == "anomaly"),
        anomaly_score=anomaly_result["anomaly_score"],
        failed_logins=event.failed_logins or 0,
        unique_dest_ports=float(df.get("unique_dest_ports_by_source", pd.Series([0])).iloc[0] or 0),
        connection_frequency=float(df.get("connection_frequency", pd.Series([0])).iloc[0] or 0),
        total_bytes=total_bytes,
        classifier_confidence=confidence,
        weights=None,
    )

    alert_id = db.insert_alert({
        "timestamp": event.timestamp,
        "source_ip": event.source_ip,
        "destination_ip": event.destination_ip,
        "destination_port": event.destination_port,
        "anomaly_score": anomaly_result["anomaly_score"],
        "prediction": anomaly_result["prediction"],
        "threat_category": threat_category,
        "confidence": confidence,
        "risk_score": risk["risk_score"],
        "severity": risk["severity"],
        "status": "open",
    })

    return AnalyzeResponse(
        alert_id=alert_id,
        prediction=anomaly_result["prediction"],
        anomaly_score=anomaly_result["anomaly_score"],
        threat_category=threat_category,
        confidence=confidence,
        risk_score=risk["risk_score"],
        severity=risk["severity"],
    )


@app.get("/alerts")
def get_alerts(severity: Optional[str] = None, threat_category: Optional[str] = None,
                status: Optional[str] = None):
    return db.list_alerts(severity=severity, threat_category=threat_category, status=status)


@app.get("/alerts/{alert_id}")
def get_alert(alert_id: int):
    alert = db.get_alert(alert_id)
    if not alert:
        raise HTTPException(404, f"Alert {alert_id} not found.")
    return alert


@app.post("/alerts/{alert_id}/explain")
def explain_alert(alert_id: int):
    alert = db.get_alert(alert_id)
    if not alert:
        raise HTTPException(404, f"Alert {alert_id} not found.")

    try:
        explanation = generate_security_explanation(alert)
    except GeminiUnavailableError as e:
        raise HTTPException(503, str(e))

    db.insert_ai_explanation(alert_id, explanation)
    return explanation


@app.get("/statistics")
def statistics():
    return db.get_statistics()
