GitHub Repository Health Predictor

A full-stack data science project that analyzes public GitHub repositories and predicts their health and risk of abandonment using machine learning, combined with rule-based development activity analysis. The system produces an interpretable health score, risk classification, and actionable recommendations through an interactive Streamlit dashboard.

Project Overview

Open-source repositories vary widely in quality, activity, and long-term sustainability. This project aims to answer the following question:

Is a GitHub repository healthy, stable, or at risk of abandonment?

To achieve this, the system:

Collects data from over 1,000 public GitHub repositories

Engineers meaningful activity and engagement features

Trains a supervised machine learning model to predict abandonment risk

Converts model outputs into a 0–100 health score

Applies rule-based logic to classify development activity

Presents results through a live, interactive dashboard

Key Features

Machine learning–based abandonment risk prediction

Health score on a 0–100 scale

Development activity classification (active, maintenance mode, low activity)

Actionable recommendations based on detected weaknesses

Live GitHub API integration for real-time analysis

Clean, modular, and reproducible ML pipeline

System Architecture
GitHub API
   ↓
Data Collection (collector.py)
   ↓
Heuristic Labeling (labeler.py)
   ↓
Feature Engineering (feature_engineering.py)
   ↓
Model Training (XGBoost)
   ↓
Prediction and Health Scoring
   ↓
Streamlit Dashboard


The system intentionally separates responsibilities:

Machine learning estimates abandonment risk

Rule-based logic interprets development activity

This avoids forcing subjective lifecycle concepts into the model itself.

Data and Features
Data

Approximately 1,000 public GitHub repositories

Balanced mix of active, archived, and inactive repositories

Data collected using the GitHub REST API

Core Features

Stars

Forks

Watchers

Open issues

Repository age (days)

Days since last push

Fork-to-star ratio

Presence of repository description

Model Details

Model: XGBoost (binary classification)

Target: At-risk versus healthy repository

Evaluation metrics:

Precision on at-risk class: approximately 97–98%

ROC-AUC: approximately 0.99

The model is optimized for high precision on identifying at-risk repositories in order to minimize false positives.

Dashboard

The Streamlit dashboard allows users to:

Enter any public GitHub repository in owner/repo format

View health score and risk probability

See development activity classification

Inspect key repository metrics

Receive actionable recommendations

All repository data is fetched live from GitHub at analysis time.

Installation and Setup
Clone the repository
git clone https://github.com/Sanjay24-05/github-health-predictor.git
cd github-health-predictor

Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

Install dependencies
pip install -r requirements.txt

Configure GitHub API access

Create a .env file based on .env.example:

GITHUB_TOKEN=your_personal_access_token


Required token scopes:

public_repo

read:org

Running the Project
Launch the dashboard
streamlit run dashboard/app.py

Reproduce the full ML pipeline (optional)
python src/data/collector.py
python src/data/labeler.py
python src/features/feature_engineering.py
python src/models/train.py


Raw datasets and trained models are intentionally excluded from version control to maintain repository cleanliness and reproducibility.

Project Structure
github-health-predictor/
├── dashboard/          # Streamlit application
├── src/
│   ├── data/           # Data collection and labeling
│   ├── features/       # Feature engineering
│   └── models/         # Model training and inference
├── notebooks/          # EDA and experimentation
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

Design Decisions

Binary classification instead of multi-class to maintain model stability

Rule-based development activity classification for interpretability

High precision optimization for at-risk detection

Exclusion of generated data and models from GitHub

Clear separation between modeling and presentation layers

Future Improvements (v2 Roadmap)

Commit velocity trend analysis

Issue close rate and pull request merge latency

Dependency health and ecosystem signals

Multi-class lifecycle prediction

Model explainability and feature attribution

Cloud deployment with persisted models

Project Significance

This project demonstrates:

End-to-end machine learning system design

Real-world data collection and preprocessing

Thoughtful feature engineering and labeling

Strong separation between ML predictions and business logic

Deployment-ready dashboard implementation

It focuses on building an interpretable and practical ML product rather than a purely academic model.

License

This project is intended for educational and portfolio use.
