import joblib
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)

from train import train_models

def evaluate_models():
    models, X_test, y_test, preprocessor = train_models()

    X_test_transformed = preprocessor.transform(X_test)

    best_model = None
    best_score = 0

    for name, model in models.items():
        y_pred = model.predict(X_test_transformed)
        y_proba = model.predict_proba(X_test_transformed)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"\n📊 {name} Evaluation")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {roc_auc:.4f}")

        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model

    joblib.dump(best_model, "models/churn_model.pkl")
    print("\n🏆 Best model saved as churn_model.pkl")

if __name__ == "__main__":
    evaluate_models()
