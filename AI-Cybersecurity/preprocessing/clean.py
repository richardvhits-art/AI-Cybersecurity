"""
Module 2: Data Preprocessing
------------------------------
Cleans raw event data and builds a reusable sklearn preprocessing pipeline
(ColumnTransformer) that is saved with Joblib and reused identically at
prediction time to avoid train/predict skew (data leakage).
"""

from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns that are identifiers/metadata, not model features.
# NOTE: "label" and "attack_cat" are both included here even though only one
# is ever used as the prediction target at a time. This is intentional --
# datasets like UNSW-NB15 provide both a binary label and a named attack
# category, and each is *derived from* the other. Whichever one is NOT
# chosen as label_col must still be excluded as a feature, or the model
# leaks the answer (e.g. training on "label" while "attack_cat" sits in
# the feature set would let the model trivially read the answer).
NON_FEATURE_COLUMNS = [
    "timestamp", "source_ip", "destination_ip", "label", "attack_cat", "id",
]


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, replace infinities, and drop fully-empty rows."""
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows that are entirely NaN (garbage rows), but keep partially-missing rows
    # -- those are handled by imputers below rather than being discarded.
    df = df.dropna(how="all")

    df.attrs["duplicates_removed"] = removed
    return df


def detect_column_types(df: pd.DataFrame, label_col: Optional[str] = "label"):
    """Split columns into numeric feature columns and categorical feature columns."""
    feature_df = df.drop(
        columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns and c != label_col],
        errors="ignore",
    )
    if label_col in feature_df.columns:
        feature_df = feature_df.drop(columns=[label_col])

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - imputes + scales numeric columns
      - imputes + one-hot encodes categorical columns
    This exact object is fit once on training data and reused (via Joblib) for prediction.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        raise ValueError("No usable numeric or categorical feature columns found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_and_transform(df: pd.DataFrame, label_col: str = "label"):
    """
    Fit a fresh preprocessing pipeline on df and transform it.
    Returns (X_transformed, preprocessor, numeric_cols, categorical_cols).
    Use this ONLY during training.
    """
    numeric_cols, categorical_cols = detect_column_types(df, label_col)
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)

    feature_df = df[numeric_cols + categorical_cols]
    X = preprocessor.fit_transform(feature_df)

    return X, preprocessor, numeric_cols, categorical_cols


def transform_with_existing(df: pd.DataFrame, preprocessor: ColumnTransformer,
                             numeric_cols: list, categorical_cols: list):
    """
    Transform new/incoming data using an ALREADY-FITTED preprocessor.
    Use this at prediction time so training and inference stay consistent.
    Missing expected columns are added as NaN so the imputer can handle them.
    """
    df = df.copy()
    for col in numeric_cols + categorical_cols:
        if col not in df.columns:
            df[col] = np.nan

    feature_df = df[numeric_cols + categorical_cols]
    return preprocessor.transform(feature_df)


def save_preprocessor(preprocessor: ColumnTransformer, numeric_cols: list,
                       categorical_cols: list, path: str = "models/preprocessor.joblib") -> str:
    payload = {
        "preprocessor": preprocessor,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
    joblib.dump(payload, path)
    return path


def load_preprocessor(path: str = "models/preprocessor.joblib"):
    payload = joblib.load(path)
    return payload["preprocessor"], payload["numeric_cols"], payload["categorical_cols"]
