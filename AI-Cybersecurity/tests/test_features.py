import pandas as pd

from preprocessing.features import engineer_features, get_available_features_report


def test_engineer_features_adds_packets_per_second():
    df = pd.DataFrame({"packet_count": [100], "flow_duration": [10]})
    result, added = engineer_features(df)
    assert "packets_per_second" in added
    assert result["packets_per_second"].iloc[0] == 10


def test_engineer_features_skips_missing_columns():
    df = pd.DataFrame({"unrelated_col": [1, 2, 3]})
    result, added = engineer_features(df)
    assert added == []


def test_available_features_report():
    df = pd.DataFrame({"packet_count": [1], "flow_duration": [1]})
    result, added = engineer_features(df)
    report = get_available_features_report(result, added)
    assert "packets_per_second" in report["available_features"]
    assert "avg_packet_size" in report["unavailable_features"]
