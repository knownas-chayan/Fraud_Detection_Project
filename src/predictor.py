"""
predictor.py
------------
Inference module: load trained models and predict on
new/unseen transactions. Returns fraud probability,
label, and anomaly scores from all three models.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_all_models() -> dict:
    """
    Load all trained model artifacts.

    Returns:
        dict with keys: xgb, iso_forest, lof, scaler
    """
    paths = {
        "xgb": MODELS_DIR / "xgb_classifier.joblib",
        "iso_forest": MODELS_DIR / "isolation_forest.joblib",
        "lof": MODELS_DIR / "lof.joblib",
    }

    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing models: {missing}. Run train.py first."
        )

    return {k: joblib.load(p) for k, p in paths.items()}


def predict_transaction(
    transaction: dict,
    models: dict,
    threshold: float = 0.5,
) -> dict:
    """
    Run a single transaction through all models.

    Args:
        transaction: Dict of feature values.
                     Keys: V1..V28, Amount, Time (or scaled_amount, scaled_time)
        models:      Dict from load_all_models().
        threshold:   XGBoost classification threshold.

    Returns:
        dict with:
            fraud_probability   : float 0–1
            is_fraud            : bool
            xgb_label           : "FRAUD" | "LEGITIMATE"
            if_anomaly          : bool
            lof_anomaly         : bool
            confidence          : "HIGH" | "MEDIUM" | "LOW"
            risk_level          : "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    """
    # Build feature vector
    row = {k: v for k, v in transaction.items()}

    # Handle Amount/Time scaling inline (simple z-score approx)
    if "Amount" in row:
        row["scaled_amount"] = row.pop("Amount")
    if "Time" in row:
        row["scaled_time"] = row.pop("Time")
    if "Class" in row:
        row.pop("Class")

    # Sort columns to match training order (V1-V28 + scaled_amount + scaled_time)
    v_cols = [f"V{i}" for i in range(1, 29)]
    ordered_cols = v_cols + ["scaled_amount", "scaled_time"]
    X = np.array([[row.get(c, 0.0) for c in ordered_cols]])

    # XGBoost
    xgb_proba = float(models["xgb"].predict_proba(X)[0, 1])
    is_fraud_xgb = xgb_proba >= threshold

    # Isolation Forest (-1 = anomaly)
    if_raw = models["iso_forest"].predict(X)[0]
    if_anomaly = if_raw == -1
    if_score = float(-models["iso_forest"].decision_function(X)[0])

    # LOF (-1 = anomaly)
    lof_raw = models["lof"].predict(X)[0]
    lof_anomaly = lof_raw == -1
    lof_score = float(-models["lof"].decision_function(X)[0])

    # Ensemble vote: XGBoost has highest weight
    votes = sum([is_fraud_xgb, if_anomaly, lof_anomaly])
    final_fraud = bool(is_fraud_xgb)  # XGBoost is primary

    # Confidence
    if xgb_proba > 0.85 or xgb_proba < 0.15:
        confidence = "HIGH"
    elif xgb_proba > 0.65 or xgb_proba < 0.35:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Risk level
    if xgb_proba >= 0.8:
        risk_level = "CRITICAL"
    elif xgb_proba >= 0.5:
        risk_level = "HIGH"
    elif xgb_proba >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "fraud_probability": round(xgb_proba, 6),
        "is_fraud": final_fraud,
        "xgb_label": "FRAUD" if final_fraud else "LEGITIMATE",
        "if_anomaly": if_anomaly,
        "if_score": round(if_score, 6),
        "lof_anomaly": lof_anomaly,
        "lof_score": round(lof_score, 6),
        "ensemble_votes": votes,
        "confidence": confidence,
        "risk_level": risk_level,
    }


def batch_predict(
    df: pd.DataFrame,
    models: dict,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Predict fraud on a batch DataFrame.

    Args:
        df:        DataFrame with same feature columns as training.
        models:    Dict from load_all_models().
        threshold: XGBoost threshold.

    Returns:
        DataFrame with additional prediction columns appended.
    """
    results = []
    for _, row in df.iterrows():
        pred = predict_transaction(row.to_dict(), models, threshold)
        results.append(pred)

    pred_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), pred_df], axis=1)
