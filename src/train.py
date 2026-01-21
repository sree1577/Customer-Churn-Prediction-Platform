import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import (
    PROCESSED_DATA_PATH,
    PREPROCESSOR_PATH,
    TARGET_COLUMN
)

MODEL_PATH = "models/churn_model.pkl"

def load_data():
    return pd.read_csv(PROCESSED_DATA_PATH)

def train_models():
    df = load_data()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    X_train_transformed = preprocessor.transform(X_train)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_transformed, y_train)
        trained_models[name] = model
        print(f"✅ Trained {name}")

    return trained_models, X_test, y_test, preprocessor

if __name__ == "__main__":
    train_models()
