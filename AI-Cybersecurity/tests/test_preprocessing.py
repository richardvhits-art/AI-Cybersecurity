import numpy as np
import pandas as pd
import pytest

from preprocessing.clean import basic_clean, detect_column_types, fit_and_transform
from preprocessing.ingest import DataIngestionError, validate_dataset


def test_basic_clean_removes_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    cleaned = basic_clean(df)
    assert len(cleaned) == 2


def test_basic_clean_replaces_infinite_values():
    # Use a second column so an infinite value doesn't make the whole row
    # empty (basic_clean only drops rows that are FULLY empty).
    df = pd.DataFrame({"a": [1, np.inf, -np.inf], "b": [10, 20, 30]})
    cleaned = basic_clean(df)
    assert cleaned["a"].isna().sum() == 2


def test_detect_column_types():
    df = pd.DataFrame({"num": [1, 2], "cat": ["x", "y"], "label": ["normal", "attack"]})
    numeric_cols, categorical_cols = detect_column_types(df)
    assert "num" in numeric_cols
    assert "cat" in categorical_cols
    assert "label" not in numeric_cols and "label" not in categorical_cols


def test_fit_and_transform_produces_matrix():
    df = pd.DataFrame({
        "bytes_sent": [100, 200, 300],
        "protocol": ["TCP", "UDP", "TCP"],
        "label": ["normal", "attack", "normal"],
    })
    X, preprocessor, numeric_cols, categorical_cols = fit_and_transform(df)
    assert X.shape[0] == 3
    assert "bytes_sent" in numeric_cols
    assert "protocol" in categorical_cols


def test_validate_dataset_raises_on_too_few_columns():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(DataIngestionError):
        validate_dataset(df, required_min_columns=2)


def test_validate_dataset_reports_missing_values():
    df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
    report = validate_dataset(df)
    assert report["missing_value_columns"]["a"] == 1
