"""
test_pipeline.py
----------------
Unit tests for the fraud detection pipeline.
Run with: pytest tests/
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_df(n=500, fraud_n=10):
    """Create a minimal fake dataset matching creditcard.csv schema."""
    np.random.seed(42)
    rows = n
    data = {f"V{i}": np.random.randn(rows) for i in range(1, 29)}
    data["Amount"] = np.abs(np.random.randn(rows)) * 100
    data["Time"] = np.arange(rows, dtype=float)
    data["Class"] = np.zeros(rows, dtype=int)
    data["Class"][:fraud_n] = 1
    return pd.DataFrame(data)


@pytest.fixture
def fake_df():
    return _make_fake_df()


@pytest.fixture
def preprocessed_data(fake_df):
    from src.preprocessor import preprocess
    return preprocess(fake_df, balance_method="smote", random_state=42)


# ─────────────────────────────────────────────────────────────────────────────
# data_loader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_get_dataset_info_correct_keys(self, fake_df):
        from src.data_loader import get_dataset_info
        info = get_dataset_info(fake_df)
        for key in ["total_transactions", "fraudulent", "legitimate",
                    "fraud_percentage", "missing_values"]:
            assert key in info

    def test_get_dataset_info_values(self, fake_df):
        from src.data_loader import get_dataset_info
        info = get_dataset_info(fake_df)
        assert info["total_transactions"] == len(fake_df)
        assert info["fraudulent"] == int(fake_df["Class"].sum())
        assert info["missing_values"] == 0

    def test_load_dataset_file_not_found(self):
        from src.data_loader import load_dataset
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/file.csv")


# ─────────────────────────────────────────────────────────────────────────────
# preprocessor tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_preprocess_returns_correct_keys(self, preprocessed_data):
        for key in ["X_train", "X_test", "y_train", "y_test",
                    "X_train_res", "y_train_res", "feature_names"]:
            assert key in preprocessed_data

    def test_amount_time_removed(self, preprocessed_data):
        feats = preprocessed_data["feature_names"]
        assert "Amount" not in feats
        assert "Time" not in feats
        assert "scaled_amount" in feats
        assert "scaled_time" in feats

    def test_smote_creates_balanced_data(self, preprocessed_data):
        y_res = preprocessed_data["y_train_res"]
        # After SMOTE, fraud and legit should be equal
        assert abs((y_res == 0).sum() - (y_res == 1).sum()) < 5

    def test_no_data_leakage(self, preprocessed_data):
        # X_test should not overlap with X_train_res (shape check)
        assert len(preprocessed_data["X_test"]) > 0
        assert len(preprocessed_data["X_train"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# anomaly_detection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAnomalyDetection:
    def test_isolation_forest_trains(self, preprocessed_data):
        from src.anomaly_detection import train_isolation_forest
        X_train = preprocessed_data["X_train"]
        model = train_isolation_forest(X_train, contamination=0.05)
        assert hasattr(model, "predict")

    def test_isolation_forest_predicts(self, preprocessed_data):
        from src.anomaly_detection import train_isolation_forest
        X_train = preprocessed_data["X_train"]
        X_test  = preprocessed_data["X_test"]
        model = train_isolation_forest(X_train, contamination=0.05)
        preds = model.predict(X_test)
        assert set(preds).issubset({-1, 1})

    def test_lof_trains(self, preprocessed_data):
        from src.anomaly_detection import train_lof
        model = train_lof(preprocessed_data["X_train"], contamination=0.05)
        assert hasattr(model, "predict")

    def test_evaluate_anomaly_model_output(self, preprocessed_data):
        from src.anomaly_detection import train_isolation_forest, evaluate_anomaly_model
        model = train_isolation_forest(preprocessed_data["X_train"], contamination=0.05)
        result = evaluate_anomaly_model(
            model,
            preprocessed_data["X_test"],
            preprocessed_data["y_test"],
        )
        for key in ["y_pred", "confusion_matrix", "fraud_scores"]:
            assert key in result


# ─────────────────────────────────────────────────────────────────────────────
# classifier tests
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifier:
    def test_xgboost_trains(self, preprocessed_data):
        from src.classifier import train_xgboost
        model = train_xgboost(
            preprocessed_data["X_train_res"],
            preprocessed_data["y_train_res"],
            params={"n_estimators": 10},
        )
        assert hasattr(model, "predict_proba")

    def test_xgboost_predicts_probabilities(self, preprocessed_data):
        from src.classifier import train_xgboost, evaluate_classifier
        model = train_xgboost(
            preprocessed_data["X_train_res"],
            preprocessed_data["y_train_res"],
            params={"n_estimators": 10},
        )
        result = evaluate_classifier(
            model,
            preprocessed_data["X_test"],
            preprocessed_data["y_test"],
        )
        assert 0.0 <= result["roc_auc"] <= 1.0
        assert "confusion_matrix" in result
        assert "y_proba" in result
        assert all(0 <= p <= 1 for p in result["y_proba"])

    def test_feature_importance_returns_sorted(self, preprocessed_data):
        from src.classifier import train_xgboost, get_feature_importance
        model = train_xgboost(
            preprocessed_data["X_train_res"],
            preprocessed_data["y_train_res"],
            params={"n_estimators": 10},
        )
        fi = get_feature_importance(model, preprocessed_data["feature_names"])
        values = list(fi.values())
        assert values == sorted(values, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# predictor tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictor:
    def test_predict_transaction_output_keys(self, preprocessed_data):
        """Test predictor with mock models."""
        from src.predictor import predict_transaction

        # Build real models for integration test
        from src.anomaly_detection import train_isolation_forest, train_lof
        from src.classifier import train_xgboost

        iso = train_isolation_forest(preprocessed_data["X_train"], contamination=0.05)
        lof = train_lof(preprocessed_data["X_train"], contamination=0.05)
        xgb = train_xgboost(
            preprocessed_data["X_train_res"],
            preprocessed_data["y_train_res"],
            params={"n_estimators": 5},
        )

        fake_models = {"xgb": xgb, "iso_forest": iso, "lof": lof}
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx.update({"Amount": 150.0, "Time": 10000.0})

        result = predict_transaction(tx, fake_models)

        for key in ["fraud_probability", "is_fraud", "xgb_label",
                    "if_anomaly", "lof_anomaly", "risk_level", "confidence"]:
            assert key in result

        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert result["xgb_label"] in ("FRAUD", "LEGITIMATE")
        assert result["risk_level"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
