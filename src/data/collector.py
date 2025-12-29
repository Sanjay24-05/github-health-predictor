# src/data/collector.py

import os
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm
from github_client import GitHubClient

RAW_PATH = "data/raw/repositories_raw.csv"

SCHEMA_COLUMNS = [
    "repo_id",
    "full_name",
    "language",
    "stars",
    "forks",
    "watchers",
    "open_issues",
    "is_archived",
    "created_at",
    "updated_at",
    "pushed_at",
    "repo_age_days",
    "days_since_last_push",
    "description",
    "has_description"
]

def compute_days(dt):
    if not dt:
        return None
    return (datetime.now(timezone.utc) - dt).days

def load_existing():
    if os.path.exists(RAW_PATH):
        df = pd.read_csv(RAW_PATH)
        return df, set(df["full_name"])
    return pd.DataFrame(columns=SCHEMA_COLUMNS), set()

def collect():
    os.makedirs("data/raw", exist_ok=True)

    client = GitHubClient()
    df_existing, seen = load_existing()

    queries = [
        ("language:python stars:>500", 300),
        ("language:javascript stars:>500", 300),
        ("archived:true stars:>100", 200),
    ]

    new_rows = []

    for query, limit in queries:
        print(f"\n🔍 Query: {query}")
        repos = client.search_repositories(query, limit)

        for repo in tqdm(repos):
            if repo.full_name in seen:
                continue

            seen.add(repo.full_name)

            new_rows.append({
                "repo_id": repo.id,
                "full_name": repo.full_name,
                "language": repo.language,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "watchers": repo.watchers_count,
                "open_issues": repo.open_issues_count,
                "is_archived": repo.archived,
                "created_at": repo.created_at.isoformat(),
                "updated_at": repo.updated_at.isoformat(),
                "pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else None,
                "repo_age_days": compute_days(repo.created_at),
                "days_since_last_push": compute_days(repo.pushed_at),
                "description": repo.description or "",
                "has_description": bool(repo.description)
            })

        # checkpoint save
        if new_rows:
            df_existing = pd.concat(
                [df_existing, pd.DataFrame(new_rows)],
                ignore_index=True
            )
            df_existing.to_csv(RAW_PATH, index=False)
            new_rows.clear()
            print("💾 Checkpoint saved")

    print("✅ Data collection complete")

if __name__ == "__main__":
    collect()
