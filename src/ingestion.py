import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/churn.csv")
PROCESSED_DATA_PATH = Path("data/processed/churn_cleaned.csv")

def load_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError("Raw data file not found")
    return pd.read_csv(RAW_DATA_PATH)

def clean_data(df):
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df

def save_data(df):
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

def run_ingestion():
    df = load_data()
    df = clean_data(df)
    save_data(df)
    print("✅ Data ingestion completed successfully")

if __name__ == "__main__":
    run_ingestion()
