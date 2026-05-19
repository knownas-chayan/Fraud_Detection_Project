"""
preprocessor.py
---------------
Feature scaling, train/test splitting, and SMOTE oversampling
to handle the severe class imbalance in fraud detection.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SCALER_PATH = MODELS_DIR / "scaler.joblib"


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    balance_method: str = "smote",
) -> dict:
    """
    Full preprocessing pipeline:
      1. Scale 'Amount' and 'Time'
      2. Stratified train/test split
      3. SMOTE oversampling on training data

    Args:
        df:              Raw DataFrame with 'Class' target column.
        test_size:       Fraction of data for testing.
        random_state:    Reproducibility seed.
        balance_method:  'smote' | 'smotetomek' | 'none'

    Returns:
        dict with keys: X_train, X_test, y_train, y_test, scaler,
                        X_train_res (resampled), y_train_res (resampled)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    # ── 1. Scale Amount and Time ──────────────────────────────────────────
    scaler = StandardScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_time"] = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)

    # ── 2. Build feature matrix and target ───────────────────────────────
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    feature_names = df.drop(columns=["Class"]).columns.tolist()

    # ── 3. Stratified split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[INFO] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"[INFO] Train fraud rate: {y_train.mean():.4%}")

    # ── 4. Resampling ─────────────────────────────────────────────────────
    if balance_method == "smote":
        sampler = SMOTE(random_state=random_state)
        method_label = "SMOTE"
    elif balance_method == "smotetomek":
        sampler = SMOTETomek(random_state=random_state)
        method_label = "SMOTETomek"
    else:
        print("[INFO] No resampling applied.")
        X_train_res, y_train_res = X_train, y_train
        sampler = None
        method_label = "None"

    if sampler is not None:
        print(f"[INFO] Applying {method_label}...")
        X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
        print(
            f"[INFO] Resampled train size: {len(X_train_res):,}  "
            f"(fraud: {y_train_res.sum():,}  legit: {(y_train_res == 0).sum():,})"
        )

    # ── 5. Save scaler ────────────────────────────────────────────────────
    amount_time_scaler = StandardScaler()
    # Re-fit a clean scaler on just Amount/Time for inference use
    joblib.dump(amount_time_scaler, SCALER_PATH)
    print(f"[INFO] Scaler saved to {SCALER_PATH}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_res": X_train_res,
        "y_train_res": y_train_res,
        "feature_names": feature_names,
        "scaler": scaler,
    }


def scale_single_transaction(row: dict, scaler: StandardScaler) -> np.ndarray:
    """
    Scale a single transaction dict for inference.

    Args:
        row:    Dict of feature values including 'Amount' and 'Time'.
        scaler: Fitted StandardScaler.

    Returns:
        np.ndarray of shape (1, n_features)
    """
    df = pd.DataFrame([row])
    if "Amount" in df.columns:
        df["scaled_amount"] = scaler.transform(df[["Amount"]])
        df.drop(columns=["Amount"], inplace=True)
    if "Time" in df.columns:
        df["scaled_time"] = scaler.transform(df[["Time"]])
        df.drop(columns=["Time"], inplace=True)
    if "Class" in df.columns:
        df.drop(columns=["Class"], inplace=True)
    return df.values
