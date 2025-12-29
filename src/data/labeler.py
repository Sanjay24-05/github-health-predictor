import os
import pandas as pd

RAW_INPUT = "data/raw/repositories_raw.csv"
LABELED_OUTPUT = "data/processed/repos_labeled.csv"


def label_repositories():
    df = pd.read_csv(RAW_INPUT)

    # Defensive checks
    required_cols = [
        "is_archived",
        "days_since_last_push",
        "open_issues",
        "repo_age_days"
    ]

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Issue pressure
    issue_pressure = df["open_issues"] / (df["repo_age_days"] + 1)

    # Final labeling logic (LOCKED)
    df["at_risk"] = (
        (df["is_archived"]) |
        (df["days_since_last_push"] >= 365) |
        (
            (df["days_since_last_push"] >= 180) &
            (issue_pressure > 0.1)
        )
    ).astype(int)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(LABELED_OUTPUT, index=False)

    print("✅ Labeling complete")
    print(df["at_risk"].value_counts(normalize=True))


if __name__ == "__main__":
    label_repositories()