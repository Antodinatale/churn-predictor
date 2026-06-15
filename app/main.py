"""
main.py
FastAPI application exposing the churn prediction model.
Loads the model once at startup; each request is stateless.
"""

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import load_model
from features import FEATURE_COLUMNS

# ── App setup ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="GitHub User Churn Predictor",
    description=(
        "Predicts whether a GitHub user is likely to have churned "
        "(stopped engaging with the platform) based on their profile signals."
    ),
    version="1.0.0",
)

# Load model once at startup — not on every request
try:
    model = load_model()
except FileNotFoundError as e:
    model = None
    _model_error = str(e)
else:
    _model_error = None


# ── Request / Response schemas ─────────────────────────────────────────────
class UserFeatures(BaseModel):
    """
    Input features for churn prediction.
    All fields correspond to engineered features from features.py.
    """
    days_inactive: float = Field(
        ..., ge=0, description="Days since last public GitHub activity"
    )
    follower_ratio: float = Field(
        ..., ge=0, description="followers / (following + 1)"
    )
    repos_per_year: float = Field(
        ..., ge=0, description="public_repos / account_age_years"
    )
    gists_per_year: float = Field(
        ..., ge=0, description="public_gists / account_age_years"
    )
    account_age_days: float = Field(
        ..., ge=1, description="Days since the GitHub account was created"
    )
    has_no_repos: int = Field(
        ..., ge=0, le=1, description="1 if user has zero public repos, else 0"
    )
    has_no_followers: int = Field(
        ..., ge=0, le=1, description="1 if user has zero followers, else 0"
    )
    profile_completeness: int = Field(
        ..., ge=0, le=3,
        description="Sum of has_bio + has_blog + has_company (0–3)"
    )


class PredictionResponse(BaseModel):
    churned: bool
    churn_probability: float
    risk_level: str


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Returns service health status. Used by Docker/cloud health checks."""
    if model is None:
        return {"status": "degraded", "detail": _model_error}
    return {"status": "ok", "model_loaded": True}


@app.get("/features")
def get_feature_list():
    """Returns the list of expected input fields and their descriptions."""
    return {
        "features": FEATURE_COLUMNS,
        "count": len(FEATURE_COLUMNS),
        "note": "All fields are required for /predict",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(user: UserFeatures):
    """
    Accepts a JSON body with GitHub-derived user features and returns:
    - churned: boolean prediction
    - churn_probability: float 0–1
    - risk_level: 'Low' / 'Medium' / 'High'
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {_model_error}",
        )

    # Build feature array in the exact order the model was trained on
    feature_values = np.array([[
        user.days_inactive,
        user.follower_ratio,
        user.repos_per_year,
        user.gists_per_year,
        user.account_age_days,
        user.has_no_repos,
        user.has_no_followers,
        user.profile_completeness,
    ]])

    pred = model.predict(feature_values)[0]
    prob = model.predict_proba(feature_values)[0][1]

    if prob < 0.4:
        risk = "Low"
    elif prob < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    return PredictionResponse(
        churned=bool(pred),
        churn_probability=round(float(prob), 3),
        risk_level=risk,
    )

# ── SVD setup at startup ───────────────────────────────────────────────────
try:
    from scipy.sparse.linalg import svds
    _raw_df = pd.read_csv("/data/raw/github_users.csv")
    from features import compute_features
    _X_items = compute_features(_raw_df).values.astype(float)
    _k = min(5, min(_X_items.shape) - 1)
    _U, _sigma, _Vt = svds(_X_items, k=_k)
    _predicted_scores = np.dot(np.dot(_U, np.diag(_sigma)), _Vt)
    _svd_ready = True
except Exception as e:
    _svd_ready = False
    _svd_error = str(e)


class RecommendRequest(BaseModel):
    user_idx: int = Field(..., ge=0, description="User index in the dataset")
    top_n: int = Field(5, ge=1, le=10)


@app.post("/recommend")
def recommend(request: RecommendRequest):
    if not _svd_ready:
        raise HTTPException(status_code=503, detail=f"SVD not ready: {_svd_error}")
    if request.user_idx >= len(_predicted_scores):
        raise HTTPException(status_code=404, detail="User index out of range")

    user_features = _X_items[request.user_idx].reshape(1, -1)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    churn_prob = model.predict_proba(user_features)[0][1]

    if churn_prob < 0.5:
        return {
            "user_idx": request.user_idx,
            "churn_probability": round(float(churn_prob), 3),
            "message": "User not at high risk — no intervention needed",
            "recommendations": []
        }

    user_scores = _predicted_scores[request.user_idx].copy()
    top_items = np.argsort(user_scores)[::-1][:request.top_n]

    return {
        "user_idx": request.user_idx,
        "churn_probability": round(float(churn_prob), 3),
        "risk_level": "High" if churn_prob >= 0.7 else "Medium",
        "recommendations": [
            {
                "similar_user_idx": int(idx),
                "score": round(float(user_scores[idx]), 3),
                "action": f"Engage with repositories from user #{idx}"
            }
            for idx in top_items
        ]
    }