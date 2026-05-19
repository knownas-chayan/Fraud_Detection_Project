"""
data_loader.py
--------------
Handles downloading and loading the Kaggle Credit Card Fraud dataset.
"""

import os
import zipfile
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "creditcard.csv"
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
CSV_DIR = Path(__file__).resolve().parent.parent / "csv"


def download_dataset():
    """
    Download the Kaggle credit card fraud dataset.
    Requires Kaggle API credentials (~/.kaggle/kaggle.json).
    """
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise ImportError("Install kaggle: pip install kaggle")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_PATH.exists():
        print(f"[INFO] Dataset already exists at {DATASET_PATH}")
        return

    print("[INFO] Downloading dataset from Kaggle...")
    os.system(
        f"kaggle datasets download -d {KAGGLE_DATASET} "
        f"-p {DATA_DIR} --unzip"
    )

    zip_path = DATA_DIR / "creditcardfraud.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()

    if DATASET_PATH.exists():
        print(f"[INFO] Dataset saved to {DATASET_PATH}")
    else:
        raise FileNotFoundError(
            "Download failed. Manually place creditcard.csv in the data/ folder."
        )


def load_dataset(path: str = None) -> pd.DataFrame:
    """
    Load the credit card fraud CSV into a DataFrame.

    Args:
        path: Optional custom path. Defaults to data/creditcard.csv.

    Returns:
        pd.DataFrame: Raw dataset.
    """
    if path:
        csv_path = Path(path)
    else:
        # Prefer the default data/creditcard.csv, otherwise look in top-level csv/ folder
        if DATASET_PATH.exists():
            csv_path = DATASET_PATH
        elif CSV_DIR.exists():
            csv_files = sorted(CSV_DIR.glob("*.csv"))
            if csv_files:
                csv_path = csv_files[0]
            else:
                raise FileNotFoundError(
                    f"No CSV files found in {CSV_DIR}.\n"
                    "Place creditcard.csv in data/ or add a CSV into csv/."
                )
        else:
            raise FileNotFoundError(
                f"Dataset not found. Checked {DATASET_PATH} and {CSV_DIR}.\n"
                "Run download_dataset() or place creditcard.csv in data/."
            )

    print(f"[INFO] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Return key statistics about the dataset.

    Args:
        df: Raw DataFrame.

    Returns:
        dict: Summary statistics.
    """
    total = len(df)
    fraud = int(df["Class"].sum())
    legit = total - fraud
    fraud_pct = fraud / total * 100

    return {
        "total_transactions": total,
        "fraudulent": fraud,
        "legitimate": legit,
        "fraud_percentage": round(fraud_pct, 4),
        "features": list(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "amount_mean": round(df["Amount"].mean(), 2),
        "amount_max": round(df["Amount"].max(), 2),
    }


if __name__ == "__main__":
    df = load_dataset()
    info = get_dataset_info(df)
    for k, v in info.items():
        print(f"  {k}: {v}")
