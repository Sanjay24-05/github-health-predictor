# src/models/train.py

import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    roc_auc_score
)

from xgboost import XGBClassifier

FEATURES_PATH = "data/features/X.csv"
LABELS_PATH = "data/features/y.csv"
MODEL_OUTPUT = "models/final_model.pkl"

def load_data():
    X = pd.read_csv(FEATURES_PATH)
    y = pd.read_csv(LABELS_PATH).values.ravel()
    return X, y

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    print(f"\n📊 {name} Results")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))

    precision = precision_score(y_test, y_pred)
    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"Precision (at-risk): {precision:.3f}")
    print(f"ROC-AUC: {roc:.3f}")

    return precision, roc

def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    scores = {}

    # -----------------------
    # Logistic Regression
    # -----------------------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    scores["Logistic Regression"] = evaluate(lr, X_test, y_test, "Logistic Regression")

    # -----------------------
    # Random Forest
    # -----------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    scores["Random Forest"] = evaluate(rf, X_test, y_test, "Random Forest")

    # -----------------------
    # XGBoost
    # -----------------------
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_train, y_train)
    scores["XGBoost"] = evaluate(xgb, X_test, y_test, "XGBoost")

    # -----------------------
    # Model selection (by precision)
    # -----------------------
    best_model_name = max(scores, key=lambda k: scores[k][0])
    best_model = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb
    }[best_model_name]

    os.makedirs("models", exist_ok=True)
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(best_model, f)

    print(f"\n🏆 Best model: {best_model_name}")
    print("✅ Saved to models/final_model.pkl")

if __name__ == "__main__":
    train()
