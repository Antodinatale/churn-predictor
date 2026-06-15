# GitHub User Churn Predictor

**Course:** Introduction to Data Science  
**Professor:** Yrupe Fresco  
**Data Source:** GitHub REST API (`api.github.com`)

**Student:** Antonella Di Natale
---

## What this project does

Predicts whether a GitHub user has churned, stopped engaging with the platform, using machine learning. It fetches real GitHub user profiles automatically, engineers behavioral features, applies four feature selection methods, trains a Random Forest model, and exposes predictions through a REST API running in Docker.

---

## How to run it (professor)

```bash
git clone <repo-url>
cd churn-predictor
docker-compose up
```

That's it. The trained model and data are already included in the repository.

Test the API:
```bash
# Health check
curl http://localhost:8000/health

# Predict churn
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "days_inactive": 250,
    "follower_ratio": 0.1,
    "repos_per_year": 0.5,
    "gists_per_year": 0.0,
    "account_age_days": 1800,
    "has_no_repos": 0,
    "has_no_followers": 0,
    "profile_completeness": 1
  }'
```

Expected response:
```json
{
  "churned": true,
  "churn_probability": 0.874,
  "risk_level": "High"
}
```

Interactive docs available at: `http://localhost:8000/docs`

---

## Project structure

```
churn-predictor/
├── app/
│   ├── main.py           # FastAPI app — /predict, /health, /features
│   ├── model.py          # Training, evaluation, model persistence
│   ├── features.py       # Feature engineering (8 features)
│   ├── scraper.py        # GitHub Search API + profile fetcher
│   └── model.pkl         # Trained model (committed, ready for Docker)
├── notebooks/
│   └── eda_and_selection.ipynb  # EDA + all 4 feature selection methods
├── data/
│   └── raw/
│       └── github_users.csv     # Fetched GitHub profiles (committed)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .gitignore
```

---

## Data collection

`scraper.py` uses the GitHub Search API with six queries (varying follower counts and 
repository counts) to automatically discover 300 real users — no hardcoded username list. 
A GitHub personal access token is used for authenticated requests (5,000 req/hr). 
It then fetches the full profile for each user and saves the result to `data/raw/github_users.csv`.

---

## Features engineered

| Feature | Type | Description |
|---|---|---|
| `days_inactive` | Time-based | Days since last public GitHub activity |
| `account_age_days` | Time-based | Days since account creation |
| `follower_ratio` | Ratio | followers / (following + 1) |
| `repos_per_year` | Aggregation | public_repos / account_age_years |
| `gists_per_year` | Aggregation | public_gists / account_age_years |
| `has_no_repos` | Binary | 1 if user has zero public repos |
| `has_no_followers` | Binary | 1 if user has zero followers |
| `profile_completeness` | Binary sum | has_bio + has_blog + has_company (0–3) |

---

## Churn label

A user is labeled **churned (1)** if their GitHub profile has not shown any public activity in more than **180 days**. See Section 2 of the report for full justification.

---

## Feature selection methods

Four methods were applied and compared in the notebook:
1. **Filter** — correlation matrix, variance threshold, ANOVA F-test
2. **Wrapper (RFE)** — recursive feature elimination with Logistic Regression
3. **Decision Tree** — single tree feature importances
4. **Random Forest** — averaged importances across 100 trees

---

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check |
| `/features` | GET | List of expected input fields |
| `/predict` | POST | Returns churn prediction + probability + risk level |
| `/docs` | GET | Interactive Swagger UI |

---

## Recreating the model (optional)

If you want to retrain from scratch:

```bash
pip install -r requirements.txt
# Open notebooks/eda_and_selection.ipynb and run all cells
# This will re-fetch GitHub data and save a new model.pkl
```
