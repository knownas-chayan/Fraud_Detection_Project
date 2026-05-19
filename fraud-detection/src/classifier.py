"""
classifier.py
-------------
XGBoost-based supervised classifier for fraud detection.
Includes training, cross-validation, hyperparameter tuning,
and model persistence.
"""

import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
XGB_MODEL_PATH = MODELS_DIR / "xgb_classifier.joblib"


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict = None,
    random_state: int = 42,
) -> XGBClassifier:
    """
    Train an XGBoost classifier.

    Args:
        X_train:      Resampled training features.
        y_train:      Resampled training labels.
        params:       Optional XGBoost hyperparameters dict.
        random_state: Seed for reproducibility.

    Returns:
        Fitted XGBClassifier.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    default_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,        # balanced after SMOTE
        "use_label_encoder": False,
        "eval_metric": "aucpr",
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    if params:
        default_params.update(params)

    print("[INFO] Training XGBoost classifier...")
    model = XGBClassifier(**default_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=50,
    )

    joblib.dump(model, XGB_MODEL_PATH)
    print(f"[INFO] XGBoost model saved to {XGB_MODEL_PATH}")
    return model


def cross_validate_model(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict:
    """
    Perform stratified k-fold cross-validation.

    Args:
        model: XGBClassifier (unfitted clone).
        X:     Full feature matrix (pre-SMOTE for fair eval).
        y:     Full labels.
        cv:    Number of folds.

    Returns:
        dict with mean/std of ROC-AUC and AP scores.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    print(f"[INFO] Running {cv}-fold cross-validation...")
    roc_scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    ap_scores = cross_val_score(
        model, X, y, cv=skf, scoring="average_precision", n_jobs=-1
    )

    print(f"  ROC-AUC:  {roc_scores.mean():.4f} ± {roc_scores.std():.4f}")
    print(f"  Avg Prec: {ap_scores.mean():.4f} ± {ap_scores.std():.4f}")

    return {
        "roc_auc_mean": roc_scores.mean(),
        "roc_auc_std": roc_scores.std(),
        "ap_mean": ap_scores.mean(),
        "ap_std": ap_scores.std(),
    }


def evaluate_classifier(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Full evaluation of the XGBoost classifier on test set.

    Args:
        model:     Fitted XGBClassifier.
        X_test:    Test features.
        y_test:    True test labels.
        threshold: Probability threshold for classification.

    Returns:
        dict with metrics and curve data.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])

    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)

    print(f"\n{'='*50}")
    print("  XGBoost Classifier Evaluation")
    print(f"{'='*50}")
    print(report)
    print(f"  ROC-AUC:           {roc_auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_curve": (fpr, tpr, roc_thresholds),
        "pr_curve": (precision, recall, pr_thresholds),
        "threshold": threshold,
    }


def get_feature_importance(
    model: XGBClassifier, feature_names: list
) -> dict:
    """
    Get feature importances from XGBoost.

    Args:
        model:         Fitted XGBClassifier.
        feature_names: List of feature names.

    Returns:
        dict mapping feature → importance, sorted descending.
    """
    importances = model.feature_importances_
    fi = dict(zip(feature_names, importances))
    return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))


def load_xgb_model() -> XGBClassifier:
    """
    Load the saved XGBoost model.

    Returns:
        Fitted XGBClassifier.
    """
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {XGB_MODEL_PATH}. Run train.py first."
        )
    model = joblib.load(XGB_MODEL_PATH)
    print(f"[INFO] XGBoost model loaded from {XGB_MODEL_PATH}")
    return model
