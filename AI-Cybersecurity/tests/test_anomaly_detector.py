import numpy as np
import pytest

from detection.anomaly_detector import AnomalyDetector


def test_predict_before_training_raises():
    detector = AnomalyDetector()
    with pytest.raises(RuntimeError):
        detector.predict(np.array([[1, 2, 3]]))


def test_train_and_predict_returns_expected_shape():
    X = np.random.RandomState(0).normal(size=(50, 3))
    detector = AnomalyDetector(n_estimators=50, contamination=0.1, random_state=0)
    detector.train(X)
    results = detector.predict(X[:5])
    assert len(results) == 5
    for r in results:
        assert r["prediction"] in ("normal", "anomaly")
        assert 0.0 <= r["anomaly_score"] <= 1.0


def test_save_untrained_model_raises(tmp_path):
    detector = AnomalyDetector()
    with pytest.raises(RuntimeError):
        detector.save(str(tmp_path / "model.joblib"))


def test_save_and_load_roundtrip(tmp_path):
    X = np.random.RandomState(1).normal(size=(30, 2))
    detector = AnomalyDetector(n_estimators=20, random_state=1)
    detector.train(X)
    path = str(tmp_path / "model.joblib")
    detector.save(path)

    loaded = AnomalyDetector.load(path)
    results = loaded.predict(X[:3])
    assert len(results) == 3
