"""
results_fetcher.py — Fetch yesterday's final WC match scores from API-Football.

Called at the start of each morning pipeline run to ingest results before
generating today's predictions.

Uses the same RAPIDAPI_KEY env var as context_fetcher.py.
Free tier: 100 calls/day — this adds 1 call per morning run.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Optional

import requests

_BASE_URL  = "https://api-football-v1.p.rapidapi.com/v3"
_API_HOST  = "api-football-v1.p.rapidapi.com"
_WC_LEAGUE = 1
_WC_SEASON = 2026
_TIMEOUT   = 15

# Match statuses that mean the final score is settled
_FINISHED_STATUSES = {"FT", "AET", "PEN"}


def fetch_yesterday_results(api_key: Optional[str] = None) -> list[dict]:
    """
    Fetch all finished WC match results from yesterday.

    Returns a list of:
        {"home_team": str, "away_team": str, "home_goals": int, "away_goals": int}

    Uses score.fulltime (the 90-minute score) so predictions are evaluated on
    regular-time outcomes, not extra time or penalty results.

    Returns empty list on any failure — the morning pipeline degrades gracefully.
    """
    api_key = api_key or os.environ.get("RAPIDAPI_KEY", "")
    if not api_key:
        print("[results] RAPIDAPI_KEY not set — skipping result ingestion.")
        return []

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    print(f"[results] Fetching finished WC results for {yesterday}...")

    try:
        resp = requests.get(
            f"{_BASE_URL}/fixtures",
            headers={
                "X-RapidAPI-Key":  api_key,
                "X-RapidAPI-Host": _API_HOST,
            },
            params={
                "league": _WC_LEAGUE,
                "season": _WC_SEASON,
                "date":   yesterday,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[results] Request failed: {exc}")
        return []

    if data.get("errors"):
        print(f"[results] API error: {data['errors']}")
        return []

    results: list[dict] = []
    for fx in data.get("response", []):
        status = fx.get("fixture", {}).get("status", {}).get("short", "")
        if status not in _FINISHED_STATUSES:
            continue

        # Prefer score.fulltime (90-min score); fall back to goals object
        ft         = fx.get("score", {}).get("fulltime", {})
        home_goals = ft.get("home")
        away_goals = ft.get("away")
        if home_goals is None or away_goals is None:
            goals      = fx.get("goals", {})
            home_goals = goals.get("home")
            away_goals = goals.get("away")
        if home_goals is None or away_goals is None:
            continue

        home_name = fx.get("teams", {}).get("home", {}).get("name", "")
        away_name = fx.get("teams", {}).get("away", {}).get("name", "")
        results.append({
            "home_team":  home_name,
            "away_team":  away_name,
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
        })
        print(f"[results]   {home_name} {home_goals}-{away_goals} {away_name}")

    print(f"[results] {len(results)} finished result(s) found.")
    return results
