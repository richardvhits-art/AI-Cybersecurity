"""
UNSW-NB15 Model Training
------------------------
Trains:
  - AnomalyDetector (Isolation Forest)  -> primary detector, trained on
    NORMAL traffic only, predicts anomaly vs normal (binary).
  - ThreatClassifier (Random Forest)    -> secondary classifier, trained
    on attack_cat, predicts a specific threat category + confidence.

Both are saved as their wrapper classes (via .save()), NOT as raw
sklearn estimators, because api/main.py loads them with
AnomalyDetector.load(...) / ThreatClassifier.load(...), which require
an instance of the wrapper class, not a bare IsolationForest /
RandomForestClassifier.

The preprocessor is built and saved through preprocessing.clean so its
saved format (keys: preprocessor / numeric_cols / categorical_cols)
matches what preprocessing.clean.load_preprocessor() expects at
inference time in api/main.py.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from detection.anomaly_detector import AnomalyDetector
from detection.classifier import ThreatClassifier
from preprocessing.clean import (
    basic_clean,
    fit_and_transform,
    save_preprocessor,
    transform_with_existing,
)

DATA_DIR = Path("data/raw/unsw_nb15")
MODELS_DIR = Path("models")

TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"


def print_binary_metrics(y_true, y_pred):
    print(f"      Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"      Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"      Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"      F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")


def main():

    print("=" * 70)
    print("AI CYBERSECURITY - UNSW-NB15 MODEL TRAINING")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------

    print("\n[1/7] Loading UNSW-NB15 datasets...")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Testing dataset not found: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"      Train rows: {len(train_df):,}  Test rows: {len(test_df):,}")

    if "label" not in train_df.columns or "label" not in test_df.columns:
        raise ValueError("Both datasets must have a 'label' column.")

    has_attack_cat = "attack_cat" in train_df.columns and "attack_cat" in test_df.columns

    # --------------------------------------------------
    # 2. Clean
    # --------------------------------------------------

    print("\n[2/7] Cleaning datasets...")

    train_df = basic_clean(train_df)
    test_df = basic_clean(test_df)

    print(f"      Train rows after cleaning: {len(train_df):,}")
    print(f"      Test rows after cleaning:  {len(test_df):,}")

    y_train_label = train_df["label"].astype(int).to_numpy()
    y_test_label = test_df["label"].astype(int).to_numpy()

    # --------------------------------------------------
    # 3. Preprocessing (fit on train, save, transform test)
    # --------------------------------------------------

    print("\n[3/7] Building preprocessing pipeline...")

    X_train, preprocessor, numeric_cols, categorical_cols = fit_and_transform(
        train_df, label_col="label"
    )
    X_test = transform_with_existing(test_df, preprocessor, numeric_cols, categorical_cols)

    print(f"      Numeric columns:     {len(numeric_cols)}")
    print(f"      Categorical columns: {len(categorical_cols)}")
    print(f"      X_train shape: {X_train.shape}")
    print(f"      X_test shape:  {X_test.shape}")

    save_preprocessor(
        preprocessor, numeric_cols, categorical_cols,
        path=str(MODELS_DIR / "preprocessor.joblib"),
    )
    print("      Saved models/preprocessor.joblib")

    # --------------------------------------------------
    # 4. Anomaly Detector (Isolation Forest, trained on normal traffic only)
    # --------------------------------------------------

    print("\n[4/7] Training AnomalyDetector...")

    normal_mask = y_train_label == 0
    X_train_normal = X_train[normal_mask]
    print(f"      Normal training samples: {X_train_normal.shape[0]:,}")

    anomaly_detector = AnomalyDetector(
        n_estimators=200,
        contamination=0.10,
        random_state=42,
    )
    anomaly_detector.train(X_train_normal)
    anomaly_detector.save(str(MODELS_DIR / "isolation_forest.joblib"))
    print("      Saved models/isolation_forest.joblib")

    print("\n      --- AnomalyDetector Evaluation (external test set) ---")
    anomaly_results = anomaly_detector.predict(X_test)
    anomaly_pred = [1 if r["prediction"] == "anomaly" else 0 for r in anomaly_results]

    print_binary_metrics(y_test_label, anomaly_pred)
    print("\n      Confusion Matrix:")
    print(confusion_matrix(y_test_label, anomaly_pred))

    # --------------------------------------------------
    # 5. Threat Classifier (Random Forest on attack_cat)
    # --------------------------------------------------

    if not has_attack_cat:
        print(
            "\n[5/7] Skipping ThreatClassifier: no 'attack_cat' column "
            "in the dataset. Only anomaly detection is available."
        )
    else:
        print("\n[5/7] Training ThreatClassifier...")

        y_train_cat = train_df["attack_cat"].astype(str).str.strip()
        y_test_cat = test_df["attack_cat"].astype(str).str.strip()

        threat_classifier = ThreatClassifier(n_estimators=200, random_state=42)
        metrics = threat_classifier.train(X_train, y_train_cat.to_numpy())

        print("\n      --- ThreatClassifier internal holdout metrics ---")
        print(f"      Accuracy:           {metrics['accuracy']:.4f}")
        print(f"      Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"      Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"      F1 (macro):         {metrics['f1_macro']:.4f}")
        print(f"      Classes: {metrics['classes']}")

        threat_classifier.save(str(MODELS_DIR / "random_forest.joblib"))
        print("\n      Saved models/random_forest.joblib")

        # --------------------------------------------------
        # 6. External test evaluation for ThreatClassifier
        # --------------------------------------------------

        print("\n[6/7] Evaluating ThreatClassifier on external test set...")

        clf_results = threat_classifier.predict(X_test)
        clf_pred = [r["threat_category"] for r in clf_results]

        ext_accuracy = accuracy_score(y_test_cat, clf_pred)
        print(f"      External accuracy: {ext_accuracy:.4f}")

    # --------------------------------------------------
    # 7. Complete
    # --------------------------------------------------

    print("\n[7/7] Training complete.")
    print("=" * 70)
    print("Saved models:")
    print("  models/preprocessor.joblib")
    print("  models/isolation_forest.joblib")
    if has_attack_cat:
        print("  models/random_forest.joblib")
    print("=" * 70)


if __name__ == "__main__":
    main()