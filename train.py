"""
train.py
--------
End-to-end training pipeline:
  1. Load dataset
  2. Preprocess & SMOTE balance
  3. Train Isolation Forest + LOF
  4. Train XGBoost classifier
  5. Evaluate all models
  6. Save all plots and artifacts
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_dataset, get_dataset_info
from src.preprocessor import preprocess
from src.anomaly_detection import (
    train_isolation_forest,
    train_lof,
    evaluate_anomaly_model,
    plot_anomaly_scores,
)
from src.classifier import (
    train_xgboost,
    evaluate_classifier,
    get_feature_importance,
)
from src.visualizer import (
    plot_class_distribution,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_feature_importance,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train fraud detection models")
    p.add_argument("--data", type=str, default=None,
                   help="Path to creditcard.csv (default: data/creditcard.csv)")
    p.add_argument("--contamination", type=float, default=0.001,
                   help="Isolation Forest / LOF contamination rate")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="XGBoost classification threshold")
    p.add_argument("--balance", type=str, default="smote",
                   choices=["smote", "smotetomek", "none"],
                   help="Class balancing strategy")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  💳  FRAUD DETECTION — TRAINING PIPELINE")
    print("=" * 60 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────
    df = load_dataset(args.data)
    info = get_dataset_info(df)
    print("\n📊 Dataset Summary:")
    for k, v in info.items():
        if k != "features":
            print(f"   {k}: {v}")

    # ── 2. Plot class distribution ────────────────────────────────────────
    plot_class_distribution(df["Class"].values, title="Original Class Distribution")
    print("\n✅ Class distribution plot saved.")

    # ── 3. Preprocess ─────────────────────────────────────────────────────
    print(f"\n⚙️  Preprocessing with {args.balance.upper()} balancing...")
    data = preprocess(df, balance_method=args.balance)

    X_train     = data["X_train"]
    X_test      = data["X_test"]
    y_train     = data["y_train"]
    y_test      = data["y_test"]
    X_train_res = data["X_train_res"]
    y_train_res = data["y_train_res"]
    features    = data["feature_names"]

    # ── 4. Anomaly detection ──────────────────────────────────────────────
    print("\n🔍 Training anomaly detection models...")
    iso_forest = train_isolation_forest(
        X_train, contamination=args.contamination
    )
    lof = train_lof(X_train, contamination=args.contamination)

    print("\n📈 Evaluating anomaly models...")
    if_results  = evaluate_anomaly_model(iso_forest, X_test, y_test, "Isolation Forest")
    lof_results = evaluate_anomaly_model(lof, X_test, y_test, "Local Outlier Factor")

    plot_anomaly_scores(
        if_results["fraud_scores"],
        lof_results["fraud_scores"],
        y_test,
    )
    print("✅ Anomaly score plots saved.")

    # ── 5. XGBoost ────────────────────────────────────────────────────────
    print("\n🚀 Training XGBoost classifier...")
    xgb_model = train_xgboost(X_train_res, y_train_res)

    print("\n📈 Evaluating XGBoost...")
    xgb_results = evaluate_classifier(xgb_model, X_test, y_test, args.threshold)

    # ── 6. Build ROC inputs for all models ───────────────────────────────
    from sklearn.metrics import roc_curve, roc_auc_score

    # For anomaly models, build pseudo-ROC from anomaly scores
    def anomaly_roc(scores, y):
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, thresh = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        return {"roc_curve": (fpr, tpr, thresh), "roc_auc": auc}

    roc_data = {
        "XGBoost": {
            "roc_curve": xgb_results["roc_curve"],
            "roc_auc":   xgb_results["roc_auc"],
        },
        "Isolation Forest": anomaly_roc(if_results["fraud_scores"], y_test),
        "LOF":              anomaly_roc(lof_results["fraud_scores"], y_test),
    }

    # ── 7. Save plots ─────────────────────────────────────────────────────
    print("\n🎨 Saving evaluation plots...")
    plot_roc_curve(roc_data)
    plot_confusion_matrix(xgb_results["confusion_matrix"], "XGBoost")
    plot_confusion_matrix(if_results["confusion_matrix"],  "Isolation Forest")
    plot_confusion_matrix(lof_results["confusion_matrix"], "LOF")
    plot_precision_recall_curve(
        xgb_results["pr_curve"],
        xgb_results["average_precision"],
    )
    fi = get_feature_importance(xgb_model, features)
    plot_feature_importance(fi, top_n=20)
    print("✅ All plots saved to assets/")

    # ── 8. Save metrics summary ───────────────────────────────────────────
    metrics = {
        "xgboost": {
            "roc_auc":           round(xgb_results["roc_auc"], 4),
            "average_precision": round(xgb_results["average_precision"], 4),
        },
        "isolation_forest": {
            "roc_auc":           round(roc_data["Isolation Forest"]["roc_auc"], 4),
        },
        "lof": {
            "roc_auc":           round(roc_data["LOF"]["roc_auc"], 4),
        },
        "dataset": {
            "total":   info["total_transactions"],
            "fraud":   info["fraudulent"],
            "legit":   info["legitimate"],
            "fraud_%": info["fraud_percentage"],
        },
    }

    metrics_path = Path("models/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Metrics saved to models/metrics.json")
    print("\n" + "=" * 60)
    print("  🎉  TRAINING COMPLETE!")
    print(f"  XGBoost ROC-AUC:    {metrics['xgboost']['roc_auc']}")
    print(f"  XGBoost Avg Prec:   {metrics['xgboost']['average_precision']}")
    print("  Run: streamlit run app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
