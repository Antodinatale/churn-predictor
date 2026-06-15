"""
scraper.py
Fetches raw user data from the GitHub REST API.
Uses the GitHub Search API to discover users automatically.
Reads GITHUB_TOKEN from .env file for higher rate limits (5000/hr vs 60/hr).
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

def _get_headers():
    headers = {'Accept': 'application/vnd.github+json'}
    if GITHUB_TOKEN:
        headers['Authorization'] = f'Bearer {GITHUB_TOKEN}'
    return headers


def search_github_users(
    total: int = 300,
    sleep_seconds: float = 0.3,
    verbose: bool = True
) -> list:
    """
    Uses GitHub Search API to discover real usernames automatically.
    With a token: sleep 0.3s between calls (5000 req/hr limit).
    Without a token: increase sleep_seconds to 1.5 (60 req/hr limit).
    """
    if GITHUB_TOKEN:
        if verbose:
            print('GitHub token found — using authenticated requests (5000 req/hr)')
    else:
        if verbose:
            print('No token found — using unauthenticated requests (60 req/hr). This will be slow.')
        sleep_seconds = max(sleep_seconds, 1.5)

    usernames = []
    seen = set()

    queries = [
        "followers:>1000",
        "followers:100..1000",
        "followers:10..100",
        "followers:1..10",
        "repos:>50",
        "repos:1..10",
    ]

    per_page = 100
    pages_per_query = max(1, total // (len(queries) * per_page) + 1)

    for query in queries:
        if len(usernames) >= total:
            break

        for page in range(1, pages_per_query + 1):
            if len(usernames) >= total:
                break

            url = "https://api.github.com/search/users"
            params = {
                "q": f"type:user {query}",
                "per_page": per_page,
                "page": page,
            }

            try:
                response = requests.get(url, params=params,
                                        headers=_get_headers(), timeout=10)

                if response.status_code == 403:
                    if verbose:
                        print('  Rate limit hit. Sleeping 60s...')
                    time.sleep(60)
                    response = requests.get(url, params=params,
                                            headers=_get_headers(), timeout=10)

                if response.status_code != 200:
                    if verbose:
                        print(f"  Search error {response.status_code} for '{query}'")
                    break

                items = response.json().get('items', [])
                if not items:
                    break

                for item in items:
                    login = item.get('login')
                    if login and login not in seen:
                        seen.add(login)
                        usernames.append(login)

                if verbose:
                    print(f"  '{query}' page {page}: +{len(items)} users (total: {len(usernames)})")

            except requests.RequestException as e:
                if verbose:
                    print(f'  Request error: {e}')
                break

            time.sleep(sleep_seconds)

    return usernames[:total]


def fetch_github_users(
    usernames: list,
    sleep_seconds: float = 0.3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetches full profile data for a list of GitHub usernames.
    """
    if not GITHUB_TOKEN:
        sleep_seconds = max(sleep_seconds, 1.0)

    records = []

    for i, username in enumerate(usernames):
        try:
            url = f"https://api.github.com/users/{username}"
            response = requests.get(url, headers=_get_headers(), timeout=10)

            if response.status_code == 404:
                if verbose:
                    print(f'  [{i+1}/{len(usernames)}] {username}: not found, skipping')
                continue
            if response.status_code == 403:
                if verbose:
                    print(f'  Rate limit hit at user {i+1}. Sleeping 60s...')
                time.sleep(60)
                response = requests.get(url, headers=_get_headers(), timeout=10)
            if response.status_code != 200:
                if verbose:
                    print(f'  [{i+1}/{len(usernames)}] {username}: HTTP {response.status_code}, skipping')
                continue

            data = response.json()
            records.append({
                'username':     data.get('login', username),
                'public_repos': data.get('public_repos', 0),
                'public_gists': data.get('public_gists', 0),
                'followers':    data.get('followers', 0),
                'following':    data.get('following', 0),
                'created_at':   data.get('created_at'),
                'updated_at':   data.get('updated_at'),
                'has_blog':     1 if data.get('blog') else 0,
                'has_bio':      1 if data.get('bio') else 0,
                'has_company':  1 if data.get('company') else 0,
                'hireable':     1 if data.get('hireable') else 0,
                'site_admin':   1 if data.get('site_admin') else 0,
            })

            if verbose:
                print(f'  [{i+1}/{len(usernames)}] {username}: OK')

        except requests.RequestException as e:
            if verbose:
                print(f'  [{i+1}/{len(usernames)}] {username}: error — {e}')
            continue

        time.sleep(sleep_seconds)

    return pd.DataFrame(records)


def label_churn(df: pd.DataFrame, threshold_days: int = 180) -> pd.DataFrame:
    df = df.copy()
    df['last_active'] = pd.to_datetime(df['updated_at'], utc=True)
    now = datetime.now(timezone.utc)
    df['days_inactive'] = (now - df['last_active']).dt.days
    df['churned'] = (df['days_inactive'] > threshold_days).astype(int)
    return df


if __name__ == '__main__':
    print('Step 1: Discovering usernames...')
    usernames = search_github_users(total=300, verbose=True)
    print(f'\nFound {len(usernames)} usernames.')
    print('\nStep 2: Fetching profiles...')
    raw_df = fetch_github_users(usernames, verbose=True)
    raw_df = label_churn(raw_df)
    os.makedirs('/data/raw', exist_ok=True)
    raw_df.to_csv('/data/raw/github_users.csv', index=False)
    print(f'\nDone. {len(raw_df)} records saved.')