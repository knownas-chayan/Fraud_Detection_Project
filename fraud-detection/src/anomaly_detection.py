"""
anomaly_detection.py
--------------------
Unsupervised anomaly detection using:
  - Isolation Forest
  - Local Outlier Factor (LOF)

These models flag suspicious transactions WITHOUT needing labels,
serving as a first-pass filter and feature enrichment layer.
"""

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.001,
    n_estimators: int = 100,
    random_state: int = 42,
) -> IsolationForest:
    """
    Train an Isolation Forest model.

    Args:
        X_train:       Training feature matrix.
        contamination: Expected fraction of outliers.
        n_estimators:  Number of trees.
        random_state:  Seed for reproducibility.

    Returns:
        Fitted IsolationForest model.
    """
    print("[INFO] Training Isolation Forest...")
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "isolation_forest.joblib")
    print("[INFO] Isolation Forest saved.")
    return model


def train_lof(
    X_train: np.ndarray,
    contamination: float = 0.001,
    n_neighbors: int = 20,
) -> LocalOutlierFactor:
    """
    Train a Local Outlier Factor model.
    Note: LOF in novelty=True mode supports predict().

    Args:
        X_train:       Training feature matrix.
        contamination: Expected fraction of outliers.
        n_neighbors:   Number of neighbors to consider.

    Returns:
        Fitted LOF model.
    """
    print("[INFO] Training Local Outlier Factor...")
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )
    model.fit(X_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "lof.joblib")
    print("[INFO] LOF model saved.")
    return model


def evaluate_anomaly_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate an anomaly detection model.
    Isolation Forest / LOF return: 1 = normal, -1 = anomaly.
    We map -1 → 1 (fraud) and 1 → 0 (legit).

    Args:
        model:       Fitted anomaly model with .predict() method.
        X_test:      Test feature matrix.
        y_test:      True binary labels (0/1).
        model_name:  Label for printing.

    Returns:
        dict with classification_report, confusion_matrix, roc_auc.
    """
    raw_preds = model.predict(X_test)
    # Map: -1 (anomaly) → 1 (fraud), 1 (normal) → 0 (legit)
    y_pred = np.where(raw_preds == -1, 1, 0)

    # Anomaly scores (lower = more anomalous for IF, negative for LOF)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # Invert so higher score = more likely fraud
        fraud_scores = -scores
    else:
        fraud_scores = y_pred.astype(float)

    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    try:
        auc = roc_auc_score(y_test, fraud_scores)
        ap = average_precision_score(y_test, fraud_scores)
        print(f"  ROC-AUC:           {auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
    except Exception:
        auc, ap = None, None

    cm = confusion_matrix(y_test, y_pred)
    return {
        "model_name": model_name,
        "y_pred": y_pred,
        "fraud_scores": fraud_scores,
        "confusion_matrix": cm,
        "roc_auc": auc,
        "average_precision": ap,
    }


def plot_anomaly_scores(
    if_scores: np.ndarray,
    lof_scores: np.ndarray,
    y_test: np.ndarray,
    save_path: str = None,
):
    """
    Plot anomaly score distributions for both models split by true label.

    Args:
        if_scores:  Isolation Forest fraud scores.
        lof_scores: LOF fraud scores.
        y_test:     True binary labels.
        save_path:  Path to save the figure.
    """
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = save_path or str(ASSETS_DIR / "anomaly_scores.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    palette = {0: "#4ade80", 1: "#f87171"}

    for ax, scores, title in zip(
        axes,
        [if_scores, lof_scores],
        ["Isolation Forest", "Local Outlier Factor"],
    ):
        ax.set_facecolor("#1e2130")
        for label, color in palette.items():
            subset = scores[y_test == label]
            ax.hist(
                subset,
                bins=60,
                alpha=0.7,
                color=color,
                label="Fraud" if label == 1 else "Legit",
                density=True,
            )
        ax.set_title(title, color="white", fontsize=13, pad=10)
        ax.set_xlabel("Anomaly Score (higher = more anomalous)", color="#aaa")
        ax.set_ylabel("Density", color="#aaa")
        ax.tick_params(colors="#888")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(facecolor="#2a2d3e", labelcolor="white")

    plt.suptitle(
        "Anomaly Score Distributions by True Label",
        color="white",
        fontsize=15,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Anomaly score plot saved to {save_path}")
    return save_path


def load_anomaly_models() -> tuple:
    """
    Load pre-trained Isolation Forest and LOF models.

    Returns:
        (IsolationForest, LocalOutlierFactor)
    """
    if_path = MODELS_DIR / "isolation_forest.joblib"
    lof_path = MODELS_DIR / "lof.joblib"

    if not if_path.exists() or not lof_path.exists():
        raise FileNotFoundError(
            "Anomaly models not found. Run train.py first."
        )

    iso_forest = joblib.load(if_path)
    lof = joblib.load(lof_path)
    return iso_forest, lof
