"""
Module 1: Data Ingestion
-------------------------
Loads security event datasets (CSV/JSON) with strict validation.
No silent failures: every problem raises a clear, human-readable error.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Columns we consider "core" for a security event. Not all datasets will
# have all of these -- feature engineering (Module 3) adapts to what's present.
KNOWN_NUMERIC_HINTS = [
    "bytes_sent", "bytes_received", "packet_count", "flow_duration",
    "failed_logins", "unique_dest_ports", "destination_port",
]


class DataIngestionError(Exception):
    """Raised when a dataset cannot be safely loaded or is structurally invalid."""


def _validate_path(file_path: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"File not found: {file_path}")
    if path.stat().st_size == 0:
        raise DataIngestionError(f"File is empty: {file_path}")
    return path


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with validation."""
    path = _validate_path(file_path)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise DataIngestionError(f"CSV file has no columns/data: {file_path}")
    except pd.errors.ParserError as e:
        raise DataIngestionError(f"CSV parsing failed for {file_path}: {e}")

    if df.shape[0] == 0:
        raise DataIngestionError(f"CSV file loaded but contains 0 rows: {file_path}")
    if df.shape[1] == 0:
        raise DataIngestionError(f"CSV file loaded but contains 0 columns: {file_path}")

    return df


def load_json(file_path: str) -> pd.DataFrame:
    """Load a JSON file (list of records or {'events': [...]})."""
    path = _validate_path(file_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DataIngestionError(f"Invalid JSON in {file_path}: {e}")

    if isinstance(data, dict) and "events" in data:
        data = data["events"]

    if not isinstance(data, list):
        raise DataIngestionError(
            "JSON must be a list of event records, or an object with an 'events' list."
        )
    if len(data) == 0:
        raise DataIngestionError("JSON file contains no event records.")

    df = pd.DataFrame(data)
    return df


def load_dataset(file_path: str) -> pd.DataFrame:
    """Dispatch loader based on file extension. CSV is the primary supported format."""
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        return load_csv(file_path)
    if ext == ".json":
        return load_json(file_path)
    raise DataIngestionError(
        f"Unsupported file type '{ext}'. Supported: .csv, .json"
    )


def validate_dataset(df: pd.DataFrame, required_min_columns: int = 2) -> dict:
    """
    Run structural validation checks and return a report.
    Does NOT raise on recoverable issues (e.g. missing values) --
    those are handled in preprocessing. Raises only on fatal structural issues.
    """
    report = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_value_columns": {},
        "infinite_value_columns": [],
        "warnings": [],
    }

    if df.shape[1] < required_min_columns:
        raise DataIngestionError(
            f"Dataset has too few columns ({df.shape[1]}). "
            f"Need at least {required_min_columns} to analyze events."
        )

    missing = df.isna().sum()
    for col, count in missing.items():
        if count > 0:
            report["missing_value_columns"][col] = int(count)

    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        if np.isinf(numeric_df[col]).any():
            report["infinite_value_columns"].append(col)

    if report["duplicate_rows"] > 0:
        report["warnings"].append(
            f"{report['duplicate_rows']} duplicate rows found (will be removed in preprocessing)."
        )

    available_hints = [c for c in KNOWN_NUMERIC_HINTS if c in df.columns]
    if len(available_hints) == 0:
        report["warnings"].append(
            "No recognized security-relevant numeric columns found. "
            "Feature engineering will have limited signal. See README for expected column format."
        )

    return report


def save_processed(df: pd.DataFrame, filename: str) -> str:
    """Save a processed dataframe to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    return str(out_path)
