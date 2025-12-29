import os
import sys
import time
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from github import Github
from github.Auth import Token

# -------------------------------------------------
# Fix Python path so `src` is importable
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.models.predict import RepoHealthPredictor

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="GitHub Repository Health Predictor",
    layout="centered"
)

# -------------------------------------------------
# Load env + clients
# -------------------------------------------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

@st.cache_resource
def load_github_client():
    if not GITHUB_TOKEN:
        st.error("GitHub token not found. Add it to .env file.")
        st.stop()
    return Github(auth=Token(GITHUB_TOKEN))

@st.cache_resource
def load_predictor():
    return RepoHealthPredictor()

# -------------------------------------------------
# Feature extraction
# -------------------------------------------------
def fetch_repo_features(repo):
    now = pd.Timestamp.utcnow()

    stars = repo.stargazers_count
    forks = repo.forks_count
    watchers = repo.watchers_count
    open_issues = repo.open_issues_count

    days_since_last_push = (now - pd.to_datetime(repo.pushed_at)).days
    repo_age_days = (now - pd.to_datetime(repo.created_at)).days

    fork_star_ratio = forks / (stars + 1)

    return {
        "stars": stars,
        "forks": forks,
        "watchers": watchers,
        "open_issues": open_issues,
        "days_since_last_push": days_since_last_push,
        "repo_age_days": repo_age_days,
        "fork_star_ratio": fork_star_ratio,
        "has_description": bool(repo.description),
    }

# -------------------------------------------------
# Development velocity (RULE-BASED)
# -------------------------------------------------
def development_state(features):
    d = features["days_since_last_push"]

    if d <= 60:
        return "Active Development"
    elif d <= 180:
        return "Maintenance Mode"
    else:
        return "Low Activity"

# -------------------------------------------------
# Final combined status (MODEL + RULES)
# -------------------------------------------------
def final_status(health_score, features):
    dev_state = development_state(features)

    if health_score < 40:
        return "🔴 Critical (High Abandonment Risk)"

    if dev_state == "Low Activity":
        return "🟠 At Risk (Low Activity)"

    if dev_state == "Maintenance Mode":
        return "🟡 Stable (Maintenance Mode)"

    return "🟢 Healthy (Actively Maintained)"

# -------------------------------------------------
# Recommendations
# -------------------------------------------------
def generate_recommendations(features):
    recs = []

    if features["days_since_last_push"] > 180:
        recs.append("Increase commit activity to demonstrate active maintenance.")

    if features["open_issues"] > 50:
        recs.append("Reduce open issue backlog to improve maintainability.")

    if features["fork_star_ratio"] < 0.05:
        recs.append("Encourage community contributions and improve onboarding.")

    if not features["has_description"]:
        recs.append("Add a clear project description and README.")

    if not recs:
        recs.append("Repository appears well-maintained. Continue current practices.")

    return recs

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("GitHub Repository Health Predictor")
st.markdown(
    """
Analyze any **public GitHub repository** and receive:
- an **ML-based abandonment risk score**
- a **development activity classification**
- **actionable recommendations**
"""
)

repo_input = st.text_input(
    "Enter GitHub repository (owner/repo)",
    placeholder="e.g. expressjs/express"
)

analyze_btn = st.button("Analyze Repository")

# -------------------------------------------------
# Main logic
# -------------------------------------------------
if analyze_btn:
    if "/" not in repo_input:
        st.error("Invalid format. Use owner/repo")
        st.stop()

    owner, repo_name = repo_input.strip().split("/")

    with st.spinner("Fetching repository data from GitHub..."):
        try:
            gh = load_github_client()
            repo = gh.get_repo(f"{owner}/{repo_name}")
        except Exception as e:
            st.error(f"Failed to fetch repository: {e}")
            st.stop()

    predictor = load_predictor()
    features = fetch_repo_features(repo)
    prediction = predictor.predict(features)

    health_score = float(prediction["health_score"])
    risk_prob = float(prediction["risk_probability"])
    status = final_status(health_score, features)
    dev_state = development_state(features)

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------
    st.subheader("📊 Repository Health Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Health Score", f"{health_score} / 100")
    col2.metric("Risk Probability", f"{risk_prob:.4f}")
    col3.metric("Final Status", status)

    st.caption(f"Development State: **{dev_state}**")

    st.divider()

    st.subheader("📈 Key Metrics")
    metrics_df = pd.DataFrame({
        "Metric": [
            "Stars",
            "Forks",
            "Open Issues",
            "Days Since Last Push",
            "Repository Age (days)",
        ],
        "Value": [
            features["stars"],
            features["forks"],
            features["open_issues"],
            features["days_since_last_push"],
            features["repo_age_days"],
        ],
    })
    st.table(metrics_df)

    st.divider()

    st.subheader("🛠️ Recommendations")
    for rec in generate_recommendations(features):
        st.write(f"- {rec}")

    st.divider()

    st.subheader("ℹ️ How to interpret this result")
    st.markdown(
        """
- **Health Score**: ML-estimated probability that the repository is *not* at risk of abandonment  
- **Development State**: Rule-based classification using recent activity  
- A repository can be **stable but not actively evolving** — this is *not* a failure state
"""
    )