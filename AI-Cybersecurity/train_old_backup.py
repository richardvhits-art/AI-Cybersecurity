"""
Module 11: Model Training
-----------------------------
Run this once (or whenever you have new data) to train and save:
  models/preprocessor.joblib
  models/isolation_forest.joblib
  models/random_forest.joblib   (only if labels are present)

Usage:
    python train.py --data data/sample/sample_events.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from detection.anomaly_detector import AnomalyDetector
from detection.classifier import ThreatClassifier
from preprocessing.clean import basic_clean, fit_and_transform, save_preprocessor
from preprocessing.features import engineer_features, get_available_features_report
from preprocessing.ingest import DataIngestionError, load_dataset, validate_dataset

MODELS_DIR = Path("models")


def main(data_path: str, label_col: str = "label") -> None:
    print(f"[1/6] Loading dataset: {data_path}")
    try:
        df = load_dataset(data_path)
        report = validate_dataset(df)
    except DataIngestionError as e:
        print(f"ERROR: Dataset validation failed. {e}")
        sys.exit(1)

    print(f"      Rows: {report['n_rows']}, Columns: {report['n_columns']}")
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"      WARNING: {w}")

    print("[2/6] Cleaning data...")
    df = basic_clean(df)
    print(f"      Duplicates removed: {df.attrs.get('duplicates_removed', 0)}")

    print("[3/6] Engineering features...")
    df, added_features = engineer_features(df)
    feat_report = get_available_features_report(df, added_features)
    print(f"      Available features: {feat_report['available_features']}")
    if feat_report["unavailable_features"]:
        print(f"      Unavailable (missing source columns): {feat_report['unavailable_features']}")

    print("[4/6] Fitting preprocessing pipeline...")
    has_labels = label_col in df.columns and df[label_col].notna().sum() > 0
    if not has_labels:
        print(f"      WARNING: label column '{label_col}' not found or empty -- "
              f"classification will be skipped.")
    X, preprocessor, numeric_cols, categorical_cols = fit_and_transform(df, label_col=label_col)
    MODELS_DIR.mkdir(exist_ok=True)
    save_preprocessor(preprocessor, numeric_cols, categorical_cols,
                       str(MODELS_DIR / "preprocessor.joblib"))
    print(f"      Numeric columns used: {numeric_cols}")
    print(f"      Categorical columns used: {categorical_cols}")

    print("[5/6] Training Isolation Forest (anomaly detection)...")
    detector = AnomalyDetector(n_estimators=200, contamination=0.15, random_state=42)
    detector.train(X)
    detector.save(str(MODELS_DIR / "isolation_forest.joblib"))
    print("      Saved models/isolation_forest.joblib")

    print("[6/6] Training Random Forest (threat classification)...")
    if has_labels:
        y = df[label_col].astype(str).to_numpy()
        classifier = ThreatClassifier(n_estimators=200, random_state=42)
        metrics = classifier.train(X, y)
        classifier.save(str(MODELS_DIR / "random_forest.joblib"))

        print("      Saved models/random_forest.joblib")
        print("\n      --- Classifier Evaluation ---")
        print(f"      Accuracy:           {metrics['accuracy']:.4f}")
        print(f"      Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"      Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"      F1 (macro):         {metrics['f1_macro']:.4f}")
        print(f"      Classes:            {metrics['classes']}")
        print("\n      Note: macro-averaged metrics are used (not raw accuracy alone) "
              "because cybersecurity datasets are typically class-imbalanced "
              "(few attacks vs many normal events). See README for details.")
        print("\n      Classification report:")
        print(metrics["classification_report"])
    else:
        print(f"      Classification model unavailable — anomaly detection only "
              f"(no '{label_col}' column with values found in dataset).")

    print("\nTraining complete. Start the API with: uvicorn api.main:app --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the cybersecurity detection models.")
    parser.add_argument("--data", default="data/sample/sample_events.csv",
                         help="Path to training CSV/JSON dataset.")
    parser.add_argument("--label-col", default="label",
                         help="Column to use as the classification target, "
                              "e.g. 'attack_cat' for UNSW-NB15 to get named "
                              "threat categories instead of binary 0/1.")
    args = parser.parse_args()
    main(args.data, label_col=args.label_col)
