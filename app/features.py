"""
features.py
Transforms raw GitHub API fields into model-ready features.
This module is imported by both the notebook (for exploration)
and model.py (for production prediction).

Feature design philosophy:
  Raw fields are ingredients. Features are hypotheses about
  what behavioral signals correlate with disengagement.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone


# ─────────────────────────────────────────────
# The exact feature columns used by the model.
# Any change here must be reflected in main.py.
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    "days_inactive",
    "follower_ratio",
    "repos_per_year",
    "gists_per_year",
    "account_age_days",
    "has_no_repos",
    "has_no_followers",
    "profile_completeness",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates all engineered features from raw GitHub fields.

    Expects columns:
        created_at, updated_at, public_repos, public_gists,
        followers, following, has_blog, has_bio, has_company

    Returns a new DataFrame with only the FEATURE_COLUMNS.
    """
    df = df.copy()

    # ── Parse timestamps ──────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    df["created_dt"] = pd.to_datetime(df["created_at"], utc=True)
    df["updated_dt"] = pd.to_datetime(df["updated_at"], utc=True)

    # ── Time-based features ───────────────────────────────────────────
    # Feature 1: days_inactive
    # Recency signal — primary churn indicator.
    # A user who hasn't had any public GitHub activity in months is
    # likely disengaged or has moved to private repos / another platform.
    df["days_inactive"] = (now - df["updated_dt"]).dt.days

    # Feature 2: account_age_days
    # How long ago the user signed up. Needed to normalize activity counts.
    df["account_age_days"] = (now - df["created_dt"]).dt.days.clip(lower=1)

    # ── Ratio features ────────────────────────────────────────────────
    # Feature 3: follower_ratio
    # followers / (following + 1) — social engagement quality.
    # Passive users who follow many people but attract no followers tend
    # toward churn. Adding 1 prevents division by zero.
    df["follower_ratio"] = df["followers"] / (df["following"] + 1)

    # ── Aggregation / normalization features ──────────────────────────
    # Feature 4: repos_per_year
    # Normalizes repository count by account age.
    # A user with 5 repos in 1 year is far more active than one with
    # 5 repos over 10 years.
    account_age_years = df["account_age_days"] / 365.25
    df["repos_per_year"] = df["public_repos"] / account_age_years.clip(lower=0.1)

    # Feature 5: gists_per_year
    # Gists are quick code snippets — high gist activity suggests
    # an engaged developer sharing work-in-progress code.
    df["gists_per_year"] = df["public_gists"] / account_age_years.clip(lower=0.1)

    # ── Binary / categorical features ─────────────────────────────────
    # Feature 6: has_no_repos
    # Users who registered but never created a repo may have signed up
    # to follow others only — a very different engagement profile.
    # Zero repos is qualitatively different from low repos.
    df["has_no_repos"] = (df["public_repos"] == 0).astype(int)

    # Feature 7: has_no_followers
    # Users with zero followers have no social anchor on the platform.
    # Research in social platforms shows that isolated users churn faster.
    df["has_no_followers"] = (df["followers"] == 0).astype(int)

    # Feature 8: profile_completeness
    # Sum of three binary profile fields (bio, blog, company).
    # A more complete profile indicates the user has invested in
    # their presence — higher investment correlates with retention.
    df["profile_completeness"] = (
        df.get("has_bio", 0) +
        df.get("has_blog", 0) +
        df.get("has_company", 0)
    )

    # Return only the modelling columns (no raw fields, no timestamps)
    return df[FEATURE_COLUMNS]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Full pipeline: raw df → (X features, y labels).
    Returns y=None if 'churned' column is not present (inference mode).
    """
    X = compute_features(df)
    y = df["churned"] if "churned" in df.columns else None

    # Fill any remaining NaNs with column medians
    X = X.fillna(X.median(numeric_only=True))

    return X, y
