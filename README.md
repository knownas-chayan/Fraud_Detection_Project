# 💳 Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)

A full end-to-end machine learning pipeline for detecting fraudulent credit card transactions using anomaly detection (Isolation Forest, LOF) and supervised learning (XGBoost), served via an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
fraud-detection/
│
├── data/                        # Raw and processed datasets
│   └── .gitkeep
│
├── models/                      # Saved model artifacts
│   └── .gitkeep
│
├── notebooks/
│   └── fraud_detection.ipynb    # Full exploratory + training notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Dataset download & loading
│   ├── preprocessor.py          # Scaling, SMOTE balancing
│   ├── anomaly_detection.py     # Isolation Forest & LOF
│   ├── classifier.py            # XGBoost training & evaluation
│   ├── visualizer.py            # ROC curve, confusion matrix plots
│   └── predictor.py             # Inference on new transactions
│
├── tests/
│   └── test_pipeline.py         # Unit tests
│
├── app.py                       # Streamlit dashboard entry point
├── train.py                     # Full training pipeline script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Get the dataset from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place `creditcard.csv` in the `data/` directory.

> **Or** use the Kaggle API:
> ```bash
> pip install kaggle
> kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
> ```

### 4. Train the Models

```bash
python train.py
```

### 5. Launch the Dashboard

```bash
streamlit run app.py
```

---

## 📊 Dataset

- **Source**: [Kaggle - ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraud Rate**: ~0.17% (highly imbalanced)
- **Features**: 30 (V1–V28 PCA features + Time + Amount)

---

## 🧠 Models Used

| Model | Type | Purpose |
|---|---|---|
| Isolation Forest | Unsupervised | Anomaly detection |
| Local Outlier Factor | Unsupervised | Anomaly detection |
| XGBoost Classifier | Supervised | Final classification |

---

## ⚖️ Handling Class Imbalance

- **SMOTE** (Synthetic Minority Oversampling Technique) is applied on the training set
- StandardScaler is used to normalize `Amount` and `Time`
- Stratified train/test split preserves fraud ratio

---

## 📈 Evaluation Metrics

- Confusion Matrix
- ROC-AUC Curve
- Precision, Recall, F1-Score
- Average Precision Score

---

## 🖥️ Dashboard Features

- Upload or enter transaction data manually
- Real-time fraud probability prediction
- Anomaly score visualization
- Model performance comparison charts
- Interactive ROC curve

---

## 🧪 Running Tests

```bash
python -m pytest tests/
```

---

## 📦 Requirements

See `requirements.txt` for full list. Key packages:
- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `imbalanced-learn` (SMOTE)
- `streamlit`
- `matplotlib`, `seaborn`, `plotly`
- `joblib`

---

## 🤝 Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgements

- [ULB Machine Learning Group](http://mlg.ulb.ac.be) for the dataset
- Kaggle community for insights
- Scikit-Learn, XGBoost, and Streamlit teams
