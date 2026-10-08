"""
UNSW-NB15 Model Training
------------------------
Train and evaluate Isolation Forest and Random Forest
using the predefined UNSW-NB15 training/testing split.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path("data/raw/unsw_nb15")
MODELS_DIR = Path("models")

TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"

NON_FEATURE_COLUMNS = [
    "label",
    "attack_cat",
    "id",
    "timestamp",
    "source_ip",
    "destination_ip",
]


def clean_dataframe(df):
    """Basic cleaning."""
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    return df


def build_preprocessor(X):
    """Build preprocessing from training data."""

    feature_columns = [
        c for c in X.columns
        if c not in NON_FEATURE_COLUMNS
    ]

    X = X[feature_columns]

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ])

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return (
        preprocessor,
        feature_columns,
        numeric_columns,
        categorical_columns,
    )


def prepare_features(df, feature_columns):
    """Match test data to training feature schema."""

    X = df.copy()

    for column in feature_columns:
        if column not in X.columns:
            X[column] = np.nan

    return X[feature_columns]


def main():

    print("=" * 70)
    print("AI CYBERSECURITY - UNSW-NB15 MODEL TRAINING")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load training data
    # --------------------------------------------------

    print("\n[1/8] Loading UNSW-NB15 training dataset...")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)

    print(f"      Rows:    {len(train_df):,}")
    print(f"      Columns: {len(train_df.columns)}")

    # --------------------------------------------------
    # 2. Load testing data
    # --------------------------------------------------

    print("\n[2/8] Loading UNSW-NB15 testing dataset...")

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_PATH}"
        )

    test_df = pd.read_csv(TEST_PATH)

    print(f"      Rows:    {len(test_df):,}")
    print(f"      Columns: {len(test_df.columns)}")

    # Safety checks
    if len(train_df) != 175341:
        print(
            f"      WARNING: expected 175,341 training rows, "
            f"found {len(train_df):,}"
        )

    if len(test_df) != 82332:
        print(
            f"      WARNING: expected 82,332 testing rows, "
            f"found {len(test_df):,}"
        )

    # --------------------------------------------------
    # 3. Clean
    # --------------------------------------------------

    print("\n[3/8] Cleaning datasets...")

    train_df = clean_dataframe(train_df)
    test_df = clean_dataframe(test_df)

    print(f"      Training rows: {len(train_df):,}")
    print(f"      Testing rows:  {len(test_df):,}")

    if "label" not in train_df.columns:
        raise ValueError("Training dataset has no 'label' column.")

    if "label" not in test_df.columns:
        raise ValueError("Testing dataset has no 'label' column.")

    # --------------------------------------------------
    # 4. Preprocessing
    # --------------------------------------------------

    print("\n[4/8] Building preprocessing pipeline...")

    y_train = train_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    X_train_raw = train_df.drop(columns=["label"])
    X_test_raw = test_df.drop(columns=["label"])

    (
        preprocessor,
        feature_columns,
        numeric_columns,
        categorical_columns,
    ) = build_preprocessor(X_train_raw)

    X_train_raw = prepare_features(
        X_train_raw,
        feature_columns,
    )

    X_test_raw = prepare_features(
        X_test_raw,
        feature_columns,
    )

    print(f"      Numeric columns:     {len(numeric_columns)}")
    print(f"      Categorical columns: {len(categorical_columns)}")

    if categorical_columns:
        print(f"      Categorical: {categorical_columns}")

    # Fit ONLY on training data
    X_train = preprocessor.fit_transform(X_train_raw)

    # Transform test using training-fitted pipeline
    X_test = preprocessor.transform(X_test_raw)

    print(f"      X_train shape: {X_train.shape}")
    print(f"      X_test shape:  {X_test.shape}")

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "feature_columns": feature_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
        },
        MODELS_DIR / "preprocessor.joblib",
    )

    print("      Saved models/preprocessor.joblib")

    # --------------------------------------------------
    # 5. Isolation Forest
    # --------------------------------------------------

    print("\n[5/8] Training Isolation Forest...")

    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]

    # FIX: X_train_normal is a scipy sparse matrix (because OneHotEncoder
    # uses sparse_output=True). len() is ambiguous on sparse matrices --
    # use .shape[0] to get the row count instead.
    print(
        f"      Normal training samples: "
        f"{X_train_normal.shape[0]:,}"
    )

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        random_state=42,
        n_jobs=-1,
    )

    isolation_forest.fit(X_train_normal)

    joblib.dump(
        isolation_forest,
        MODELS_DIR / "isolation_forest.joblib",
    )

    print("      Saved models/isolation_forest.joblib")

    anomaly_raw = isolation_forest.predict(X_test)

    anomaly_pred = np.where(
        anomaly_raw == -1,
        1,
        0,
    )

    print("\n      --- Isolation Forest Evaluation ---")
    print(
        f"      Accuracy:  "
        f"{accuracy_score(y_test, anomaly_pred):.4f}"
    )
    print(
        f"      Precision: "
        f"{precision_score(y_test, anomaly_pred, zero_division=0):.4f}"
    )
    print(
        f"      Recall:    "
        f"{recall_score(y_test, anomaly_pred, zero_division=0):.4f}"
    )
    print(
        f"      F1:        "
        f"{f1_score(y_test, anomaly_pred, zero_division=0):.4f}"
    )

    print("\n      Confusion Matrix:")
    print(confusion_matrix(y_test, anomaly_pred))

    # --------------------------------------------------
    # 6. Random Forest
    # --------------------------------------------------

    print("\n[6/8] Training Random Forest...")

    random_forest = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    random_forest.fit(X_train, y_train)

    joblib.dump(
        random_forest,
        MODELS_DIR / "random_forest.joblib",
    )

    print("      Saved models/random_forest.joblib")

    # --------------------------------------------------
    # 7. External test evaluation
    # --------------------------------------------------

    print(
        "\n[7/8] Evaluating Random Forest "
        "on external test set..."
    )

    rf_pred = random_forest.predict(X_test)

    print("\n      --- Random Forest Evaluation ---")

    print(
        f"      Accuracy:          "
        f"{accuracy_score(y_test, rf_pred):.4f}"
    )

    print(
        f"      Precision (macro): "
        f"{precision_score(y_test, rf_pred, average='macro', zero_division=0):.4f}"
    )

    print(
        f"      Recall (macro):    "
        f"{recall_score(y_test, rf_pred, average='macro', zero_division=0):.4f}"
    )

    print(
        f"      F1 (macro):        "
        f"{f1_score(y_test, rf_pred, average='macro', zero_division=0):.4f}"
    )

    print("\n      Confusion Matrix:")
    print(confusion_matrix(y_test, rf_pred))

    print("\n      Classification Report:")
    print(
        classification_report(
            y_test,
            rf_pred,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # 8. Complete
    # --------------------------------------------------

    print("\n[8/8] Training complete.")
    print("=" * 70)
    print("Saved models:")
    print("  models/preprocessor.joblib")
    print("  models/isolation_forest.joblib")
    print("  models/random_forest.joblib")
    print("=" * 70)


if __name__ == "__main__":
    main()