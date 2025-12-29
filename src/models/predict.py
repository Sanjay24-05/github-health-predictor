import pickle
import numpy as np
import pandas as pd

MODEL_PATH = "models/final_model.pkl"
SCALER_PATH = "data/features/scaler.pkl"

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


class RepoHealthPredictor:
    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        # numeric columns that were scaled
        self.numeric_cols = [
            "stars",
            "forks",
            "watchers",
            "open_issues",
            "days_since_last_push",
            "repo_age_days",
            "fork_star_ratio",
        ]

    def _prepare_features(self, repo_features: dict) -> pd.DataFrame:
        """
        Convert raw repo features into model-ready dataframe
        """
        df = pd.DataFrame([repo_features])

        # Ensure column order
        df = df[FEATURE_COLUMNS]

        # Scale numeric features
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])

        return df

    def predict(self, repo_features: dict) -> dict:
        """
        Returns:
        - at_risk (0/1)
        - risk_probability (0-1)
        - health_score (0-100)
        """

        X = self._prepare_features(repo_features)

        risk_prob = self.model.predict_proba(X)[0, 1]
        at_risk = int(risk_prob >= 0.5)

        health_score = round((1 - risk_prob) * 100, 2)

        return {
            "at_risk": at_risk,
            "risk_probability": round(risk_prob, 4),
            "health_score": health_score,
        }


# ------------------------
# Example usage (for testing)
# ------------------------
if __name__ == "__main__":
    predictor = RepoHealthPredictor()

    example_repo = {
        "stars": 1500,
        "forks": 300,
        "watchers": 1500,
        "open_issues": 25,
        "days_since_last_push": 40,
        "repo_age_days": 1500,
        "fork_star_ratio": 300 / (1500 + 1),
        "has_description": True,
    }

    result = predictor.predict(example_repo)
    print(result)