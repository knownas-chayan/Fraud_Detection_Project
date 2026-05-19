"""
app.py
------
Streamlit dashboard for Credit Card Fraud Detection.
Features:
  - Dataset statistics overview
  - Model performance charts (ROC, confusion matrix, feature importance)
  - Manual transaction input → real-time prediction
  - CSV batch upload → bulk prediction with download
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Main background */
    .main { background-color: #0d1117; }
    .block-container { padding: 2rem 2.5rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #2d3148;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    /* Fraud badge */
    .fraud-badge {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #f87171;
        color: #fca5a5;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        display: inline-block;
    }
    .legit-badge {
        background: linear-gradient(135deg, #14532d, #166534);
        border: 1px solid #4ade80;
        color: #86efac;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Section headers */
    h1, h2, h3 { color: #e2e8f0 !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161b27;
        border-right: 1px solid #2d3148;
    }

    /* Dividers */
    hr { border-color: #2d3148; }

    /* Input labels */
    label { color: #94a3b8 !important; font-size: 0.85rem !important; }

    /* Streamlit metric */
    [data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
ASSETS = Path(__file__).parent / "assets"
MODELS = Path(__file__).parent / "models"


@st.cache_resource(show_spinner=False)
def load_models():
    """Load all trained models (cached)."""
    try:
        from src.predictor import load_all_models
        return load_all_models(), None
    except FileNotFoundError as e:
        return None, str(e)


def load_metrics():
    p = MODELS / "metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def asset_img(name):
    p = ASSETS / name
    return str(p) if p.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Fraud Detector")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔍 Predict Transaction", "📂 Batch Prediction", "📊 Model Insights"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probability ≥ threshold → FRAUD",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#475569'>Built with XGBoost + Isolation Forest + LOF<br>"
        "Dataset: Kaggle ULB Credit Card Fraud</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────
models, model_error = load_models()
metrics = load_metrics()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("💳 Credit Card Fraud Detection")
    st.markdown(
        "<p style='color:#64748b'>End-to-end ML pipeline · Isolation Forest · LOF · XGBoost</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Model status
    if model_error:
        st.warning(
            f"⚠️ Models not trained yet. Run `python train.py` first.\n\n`{model_error}`"
        )
    else:
        st.success("✅ All models loaded and ready.")

    # Dataset stats
    st.subheader("📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    ds = metrics["dataset"] if metrics else {}
    c1.metric("Total Transactions", f"{ds.get('total', 284807):,}")
    c2.metric("Fraudulent",         f"{ds.get('fraud', 492):,}")
    c3.metric("Legitimate",         f"{ds.get('legit', 284315):,}")
    c4.metric("Fraud Rate",         f"{ds.get('fraud_%', 0.173):.3f}%")

    # Model metrics
    if metrics:
        st.subheader("🏆 Model Performance (Test Set)")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("XGBoost ROC-AUC",    metrics["xgboost"].get("roc_auc", "—"))
        mc2.metric("XGBoost Avg Prec",   metrics["xgboost"].get("average_precision", "—"))
        mc3.metric("Isolation Forest AUC", metrics["isolation_forest"].get("roc_auc", "—"))

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        img = asset_img("class_distribution.png")
        if img:
            st.image(img, caption="Class Distribution", use_container_width=True)
        else:
            # Placeholder chart
            fig = go.Figure(go.Bar(
                x=["Legitimate", "Fraudulent"],
                y=[284315, 492],
                marker_color=["#4ade80", "#f87171"],
            ))
            fig.update_layout(
                title="Class Distribution",
                plot_bgcolor="#1e2130",
                paper_bgcolor="#1e2130",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        img = asset_img("roc_curve.png")
        if img:
            st.image(img, caption="ROC Curves", use_container_width=True)
        else:
            st.info("Train models to see ROC curve: `python train.py`")

    # Architecture
    st.subheader("🧠 Pipeline Architecture")
    st.markdown(
        """
| Step | Component | Description |
|------|-----------|-------------|
| 1 | **Data Loading** | Kaggle Credit Card CSV (284,807 rows) |
| 2 | **Preprocessing** | StandardScaler on Amount/Time |
| 3 | **Balancing** | SMOTE oversampling on training set |
| 4 | **Anomaly Detection** | Isolation Forest + Local Outlier Factor |
| 5 | **Classification** | XGBoost with 300 estimators |
| 6 | **Evaluation** | ROC-AUC, Confusion Matrix, Avg Precision |
| 7 | **Serving** | This Streamlit dashboard |
"""
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT TRANSACTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Predict Transaction":
    st.title("🔍 Predict a Transaction")
    st.markdown(
        "<p style='color:#64748b'>Enter transaction features to get a real-time fraud prediction.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if model_error:
        st.error(f"Models not loaded. Run `python train.py` first.")
        st.stop()

    with st.form("transaction_form"):
        st.subheader("Transaction Features")

        # Amount & Time
        col_a, col_b = st.columns(2)
        amount = col_a.number_input("Amount ($)", min_value=0.0, max_value=25000.0,
                                     value=100.0, step=0.01)
        time_val = col_b.number_input("Time (seconds from first tx)",
                                       min_value=0.0, max_value=200000.0, value=50000.0)

        st.markdown("**PCA Features (V1 – V28)**")
        st.caption("Leave at 0.0 for average transaction. Adjust to simulate anomalies.")

        # V1-V28 in a grid
        v_values = {}
        cols_per_row = 7
        v_cols_all = [f"V{i}" for i in range(1, 29)]
        rows = [v_cols_all[i:i+cols_per_row] for i in range(0, 28, cols_per_row)]

        for row_features in rows:
            cols = st.columns(len(row_features))
            for col, feat in zip(cols, row_features):
                v_values[feat] = col.number_input(feat, value=0.0, step=0.1,
                                                   format="%.3f", label_visibility="visible")

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        transaction = {**v_values, "Amount": amount, "Time": time_val}

        with st.spinner("Analyzing transaction..."):
            from src.predictor import predict_transaction
            time.sleep(0.4)
            result = predict_transaction(transaction, models, threshold)

        st.markdown("---")
        st.subheader("Prediction Result")

        col_r1, col_r2, col_r3 = st.columns([2, 1, 1])

        with col_r1:
            badge = (
                '<div class="fraud-badge">🚨 FRAUDULENT TRANSACTION</div>'
                if result["is_fraud"]
                else '<div class="legit-badge">✅ LEGITIMATE TRANSACTION</div>'
            )
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown(f"**Risk Level:** `{result['risk_level']}`")
            st.markdown(f"**Confidence:** `{result['confidence']}`")

        with col_r2:
            st.metric("Fraud Probability",
                      f"{result['fraud_probability']:.2%}",
                      delta=None)
            st.metric("Ensemble Votes",
                      f"{result['ensemble_votes']}/3 say FRAUD")

        with col_r3:
            if_status = "🔴 Anomaly" if result["if_anomaly"] else "🟢 Normal"
            lof_status = "🔴 Anomaly" if result["lof_anomaly"] else "🟢 Normal"
            st.markdown(f"**Isolation Forest:** {if_status}")
            st.markdown(f"**LOF:** {lof_status}")
            st.markdown(f"**IF Score:** `{result['if_score']:.4f}`")
            st.markdown(f"**LOF Score:** `{result['lof_score']:.4f}`")

        # Gauge chart
        prob = result["fraud_probability"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"color": "#f87171" if prob > 0.5 else "#4ade80"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569"},
                "bar": {"color": "#f87171" if prob > 0.5 else "#4ade80"},
                "bgcolor": "#1e2130",
                "steps": [
                    {"range": [0, 30],  "color": "#14532d"},
                    {"range": [30, 50], "color": "#713f12"},
                    {"range": [50, 80], "color": "#7c2d12"},
                    {"range": [80, 100], "color": "#450a0a"},
                ],
                "threshold": {
                    "line": {"color": "#fbbf24", "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
            title={"text": "Fraud Probability Gauge", "font": {"color": "#94a3b8"}},
        ))
        fig.update_layout(
            height=280,
            paper_bgcolor="#0d1117",
            font_color="#e2e8f0",
            margin=dict(t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Full Prediction Details"):
            st.json(result)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BATCH PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Transaction Prediction")
    st.markdown(
        "<p style='color:#64748b'>Upload a CSV of transactions to get bulk predictions.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if model_error:
        st.error("Models not loaded. Run `python train.py` first.")
        st.stop()

    st.info(
        "Upload a CSV with columns: **V1–V28, Amount, Time**  \n"
        "Optionally include **Class** (0/1) for accuracy evaluation."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded:
        df_input = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_input):,} rows.")

        with st.expander("Preview Data"):
            st.dataframe(df_input.head(10), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            from src.predictor import batch_predict
            with st.spinner(f"Predicting {len(df_input):,} transactions..."):
                results_df = batch_predict(df_input, models, threshold)

            fraud_count = results_df["is_fraud"].sum()
            total = len(results_df)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Transactions", f"{total:,}")
            c2.metric("Flagged as Fraud",   f"{fraud_count:,}")
            c3.metric("Fraud Rate",         f"{fraud_count/total:.2%}")

            # Distribution chart
            fig = px.histogram(
                results_df,
                x="fraud_probability",
                nbins=50,
                color_discrete_sequence=["#60a5fa"],
                title="Fraud Probability Distribution",
            )
            fig.update_layout(
                plot_bgcolor="#1e2130",
                paper_bgcolor="#1e2130",
                font_color="#e2e8f0",
                xaxis_title="Fraud Probability",
                yaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Results table (top fraud cases)
            st.subheader("🔴 Top Flagged Transactions")
            fraud_df = results_df[results_df["is_fraud"]].sort_values(
                "fraud_probability", ascending=False
            ).head(20)
            st.dataframe(
                fraud_df[["fraud_probability", "risk_level", "confidence",
                           "if_anomaly", "lof_anomaly", "ensemble_votes"]],
                use_container_width=True,
            )

            # Download
            csv_out = results_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Full Results CSV",
                data=csv_out,
                file_name="fraud_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Insights":
    st.title("📊 Model Insights & Performance")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["ROC Curve", "Confusion Matrix", "Feature Importance", "Anomaly Scores"]
    )

    with tab1:
        img = asset_img("roc_curve.png")
        if img:
            st.image(img, caption="ROC Curve Comparison", use_container_width=True)
        else:
            st.info("Train models first: `python train.py`")

        img2 = asset_img("precision_recall_curve.png")
        if img2:
            st.image(img2, caption="Precision-Recall Curve (XGBoost)", use_container_width=True)

    with tab2:
        cm_imgs = [
            ("XGBoost",          asset_img("confusion_matrix_xgboost.png")),
            ("Isolation Forest", asset_img("confusion_matrix_isolation_forest.png")),
            ("LOF",              asset_img("confusion_matrix_lof.png")),
        ]
        for name, img in cm_imgs:
            if img:
                st.image(img, caption=f"Confusion Matrix — {name}", use_container_width=True)
            else:
                st.info(f"Run `python train.py` to generate {name} confusion matrix.")

    with tab3:
        img = asset_img("feature_importance.png")
        if img:
            st.image(img, caption="XGBoost Feature Importances", use_container_width=True)
        else:
            st.info("Train models first: `python train.py`")

    with tab4:
        img = asset_img("anomaly_scores.png")
        if img:
            st.image(img, caption="Anomaly Score Distributions", use_container_width=True)
        else:
            st.info("Train models first: `python train.py`")

    if metrics:
        st.markdown("---")
        st.subheader("📋 Saved Metrics")
        st.json(metrics)
