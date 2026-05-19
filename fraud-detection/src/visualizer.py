"""
visualizer.py
-------------
All plotting functions: ROC curves, confusion matrices,
precision-recall curves, feature importance, and class distribution.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

# ── Dark theme palette ────────────────────────────────────────────────────────
BG_COLOR = "#0f1117"
PANEL_COLOR = "#1e2130"
ACCENT = "#60a5fa"
FRAUD_COLOR = "#f87171"
LEGIT_COLOR = "#4ade80"
TEXT_COLOR = "#e2e8f0"
MUTED_COLOR = "#94a3b8"


def _setup_dark_fig(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=MUTED_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3148")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    return fig, ax


def plot_class_distribution(y: np.ndarray, title: str = "Class Distribution",
                             save: bool = True) -> str:
    """Bar chart showing Legit vs Fraud count."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["Legitimate", "Fraudulent"]
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    colors = [LEGIT_COLOR, FRAUD_COLOR]

    fig, ax = _setup_dark_fig(figsize=(7, 5))
    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="none")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count:,}\n({count/sum(counts):.2%})",
            ha="center", va="bottom", color=TEXT_COLOR, fontsize=11
        )
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    path = str(ASSETS_DIR / "class_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def plot_roc_curve(
    results_dict: dict,
    save: bool = True,
) -> str:
    """
    Plot ROC curves for one or multiple models.

    Args:
        results_dict: {model_name: {"roc_curve": (fpr, tpr), "roc_auc": float}}
        save:         Whether to save the figure.

    Returns:
        Path to saved figure.
    """
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = _setup_dark_fig(figsize=(8, 6))

    colors = [ACCENT, "#f472b6", "#a78bfa", "#34d399"]
    ax.plot([0, 1], [0, 1], ":", color="#475569", linewidth=1.5, label="Random (AUC = 0.50)")

    for (name, result), color in zip(results_dict.items(), colors):
        fpr, tpr, _ = result["roc_curve"]
        auc = result["roc_auc"]
        ax.plot(fpr, tpr, linewidth=2.5, color=color, label=f"{name}  (AUC = {auc:.4f})")

    ax.fill_between(
        results_dict[list(results_dict.keys())[0]]["roc_curve"][0],
        results_dict[list(results_dict.keys())[0]]["roc_curve"][1],
        alpha=0.08, color=ACCENT,
    )
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, pad=15)
    legend = ax.legend(loc="lower right", facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR,
                       edgecolor="#2d3148", fontsize=10)
    ax.grid(True, color="#2d3148", linestyle="--", linewidth=0.8, alpha=0.6)

    path = str(ASSETS_DIR / "roc_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "XGBoost",
    save: bool = True,
) -> str:
    """
    Plot a styled confusion matrix.

    Args:
        cm:         2x2 confusion matrix array.
        model_name: Title label.
        save:       Whether to save the figure.

    Returns:
        Path to saved figure.
    """
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
        linewidths=0.5,
        linecolor="#2d3148",
        annot_kws={"size": 16, "color": "white", "weight": "bold"},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel("True Label", color=TEXT_COLOR, fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", color=TEXT_COLOR, fontsize=13, pad=15)
    ax.tick_params(colors=MUTED_COLOR, labelsize=11)

    # Color TP/TN green, FP/FN red cells
    ax.get_children()[0].set_cmap("Blues")

    path = str(ASSETS_DIR / f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def plot_precision_recall_curve(
    pr_curve: tuple,
    ap_score: float,
    model_name: str = "XGBoost",
    save: bool = True,
) -> str:
    """
    Plot precision-recall curve.

    Args:
        pr_curve:   (precision, recall, thresholds) tuple.
        ap_score:   Average Precision score.
        model_name: Title label.
        save:       Whether to save the figure.

    Returns:
        Path to saved figure.
    """
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = pr_curve

    fig, ax = _setup_dark_fig(figsize=(8, 6))
    ax.plot(recall, precision, color=ACCENT, linewidth=2.5,
            label=f"AP = {ap_score:.4f}")
    ax.fill_between(recall, precision, alpha=0.1, color=ACCENT)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=14, pad=15)
    ax.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR, edgecolor="#2d3148", fontsize=11)
    ax.grid(True, color="#2d3148", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlim([0, 1.01])
    ax.set_ylim([0, 1.05])

    path = str(ASSETS_DIR / "precision_recall_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def plot_feature_importance(
    feature_importance: dict,
    top_n: int = 20,
    save: bool = True,
) -> str:
    """
    Horizontal bar chart of top-N feature importances.

    Args:
        feature_importance: {feature_name: importance_value}
        top_n:              Number of top features to display.
        save:               Whether to save the figure.

    Returns:
        Path to saved figure.
    """
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    items = list(feature_importance.items())[:top_n]
    names = [i[0] for i in items][::-1]
    values = [i[1] for i in items][::-1]

    fig, ax = _setup_dark_fig(figsize=(9, 7))
    colors = [ACCENT if v > np.median(values) else "#4b5563" for v in values]
    bars = ax.barh(names, values, color=colors, edgecolor="none", height=0.7)

    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color=MUTED_COLOR, fontsize=9)

    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances (XGBoost)", fontsize=14, pad=15)
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(True, axis="x", color="#2d3148", linestyle="--", linewidth=0.8, alpha=0.6)

    path = str(ASSETS_DIR / "feature_importance.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path
