
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_INPUT = "data/processed/repos_labeled.csv"
FEATURE_OUTPUT = "data/features/X.csv"
LABEL_OUTPUT = "data/features/y.csv"
SCALER_OUTPUT = "data/features/scaler.pkl"

FEATURE_COLUMNS = [
    "stars",
    "forks",
    "watchers",
    "open_issues",
    "days_since_last_push",
    "repo_age_days",
    "fork_star_ratio",
    "has_description",
]

LABEL_COLUMN = "at_risk"


def engineer_features():
    # Load labeled data
    df = pd.read_csv(RAW_INPUT)

    # Safety checks
    missing = set(FEATURE_COLUMNS + [LABEL_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Feature matrix & label
    X = df[FEATURE_COLUMNS].copy()
    y = df[LABEL_COLUMN].copy()

    # Handle missing values (defensive)
    X.fillna(0, inplace=True)

    # Scale numeric features
    numeric_cols = [
        "stars",
        "forks",
        "watchers",
        "open_issues",
        "days_since_last_push",
        "repo_age_days",
        "fork_star_ratio",
    ]

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Persist outputs
    os.makedirs("data/features", exist_ok=True)

    X.to_csv(FEATURE_OUTPUT, index=False)
    y.to_csv(LABEL_OUTPUT, index=False)

    # Save scaler for inference
    import pickle
    with open(SCALER_OUTPUT, "wb") as f:
        pickle.dump(scaler, f)

    print("✅ Feature engineering complete")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


if __name__ == "__main__":
    engineer_features()
