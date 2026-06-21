"""
odds_fetcher.py — Automated odds via The Odds API (the-odds-api.com).

Free tier: 500 requests/month. One request per day = ~31/month (well within limit).
Single request fetches h2h (1X2) + totals (O/U) for all upcoming WC matches.

Returns same dict shape as the old manual get_todays_manual_odds():
  {(home_team, away_team): {"odds_1x2": MatchOdds1X2, "ou_odds": OverUnderOdds|None, "stage": TournamentStage}}

Register free at: https://the-odds-api.com
Add your key as GitHub Secret: THE_ODDS_API_KEY
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

from config.scoring_rules import TournamentStage
from core.odds_converter import MatchOdds1X2, OverUnderOdds

_ODDS_API_ENDPOINT = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds/"
_PREFERRED_REGIONS = "eu"   # European bookmakers use decimal odds natively

# ---------------------------------------------------------------------------
# WC 2026 stage date ranges (UTC). Hardcoded — tournament dates are fixed.
# ---------------------------------------------------------------------------
_STAGE_DATE_RANGES: list[tuple[datetime, datetime, TournamentStage]] = [
    (datetime(2026, 6, 11, tzinfo=timezone.utc), datetime(2026, 7,  7, tzinfo=timezone.utc), TournamentStage.GROUP_STAGE),
    (datetime(2026, 7,  7, tzinfo=timezone.utc), datetime(2026, 7, 12, tzinfo=timezone.utc), TournamentStage.ROUND_OF_32),
    (datetime(2026, 7, 12, tzinfo=timezone.utc), datetime(2026, 7, 16, tzinfo=timezone.utc), TournamentStage.ROUND_OF_16),
    (datetime(2026, 7, 16, tzinfo=timezone.utc), datetime(2026, 7, 20, tzinfo=timezone.utc), TournamentStage.QUARTER_FINAL),
    (datetime(2026, 7, 20, tzinfo=timezone.utc), datetime(2026, 7, 24, tzinfo=timezone.utc), TournamentStage.SEMI_FINAL),
    (datetime(2026, 7, 24, tzinfo=timezone.utc), datetime(2026, 7, 27, tzinfo=timezone.utc), TournamentStage.THIRD_PLACE),
    (datetime(2026, 7, 27, tzinfo=timezone.utc), datetime(2026, 7, 29, tzinfo=timezone.utc), TournamentStage.FINAL),
]

# ---------------------------------------------------------------------------
# Team name normalization: The Odds API name → canonical name
# (canonical = what football-data.org and our schedule use)
# ---------------------------------------------------------------------------
_TEAM_NAME_MAP: dict[str, str] = {
    "south korea":           "korea republic",
    "republic of korea":     "korea republic",
    "ir iran":               "iran",
    "côte d'ivoire":         "ivory coast",
    "cote d'ivoire":         "ivory coast",
    "usa":                   "united states",
    "us":                    "united states",
    "trinidad & tobago":     "trinidad and tobago",
    "czechia":               "czech republic",
    "türkiye":               "turkey",
    "turkiye":               "turkey",
    "bosnia & herzegovina":  "bosnia and herzegovina",
    "north macedonia":       "north macedonia",
    "cape verde":            "cape verde islands",
    "democratic republic of congo": "dr congo",
}


def _normalize(name: str) -> str:
    """Lowercase + strip + apply known alias map."""
    n = name.lower().strip()
    return _TEAM_NAME_MAP.get(n, n)


def _infer_stage(commence_time: datetime) -> TournamentStage:
    """Derive TournamentStage from match start time using WC 2026 date ranges."""
    for start, end, stage in _STAGE_DATE_RANGES:
        if start <= commence_time < end:
            return stage
    # Outside known ranges → best guess
    if commence_time < _STAGE_DATE_RANGES[0][0]:
        return TournamentStage.GROUP_STAGE
    return TournamentStage.FINAL


def _parse_commence_time(raw: str) -> datetime:
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _extract_odds_from_bookmaker(bookmaker: dict) -> tuple[Optional[MatchOdds1X2], Optional[OverUnderOdds]]:
    """
    Parse h2h and totals markets from a single bookmaker entry.
    Returns (MatchOdds1X2|None, OverUnderOdds|None).
    """
    odds_1x2:  Optional[MatchOdds1X2]    = None
    ou_odds:   Optional[OverUnderOdds]   = None

    home_team_name = None  # resolved once we know the match

    for market in bookmaker.get("markets", []):
        key = market.get("key", "")
        outcomes = market.get("outcomes", [])

        if key == "h2h" and len(outcomes) >= 2:
            # Outcomes: home team, away team (or "Draw")
            home_price = draw_price = away_price = None
            for o in outcomes:
                oname = o.get("name", "").lower()
                price = o.get("price", 0.0)
                if oname == "draw":
                    draw_price = price
                elif home_team_name is None or oname == _normalize(home_team_name or ""):
                    # First non-draw = home (API returns home first)
                    if home_price is None:
                        home_price = price
                    else:
                        away_price = price
                else:
                    away_price = price

            # Simpler approach: outcomes order is [home, away] or [home, draw, away]
            named = {o["name"].lower(): o["price"] for o in outcomes}
            prices = [(o["name"], o["price"]) for o in outcomes]

            if "draw" in named and len(prices) == 3:
                # Classic 1X2: find home and away around draw
                non_draw = [(n, p) for n, p in prices if n.lower() != "draw"]
                if len(non_draw) == 2:
                    odds_1x2 = MatchOdds1X2(
                        home=non_draw[0][1],
                        draw=named["draw"],
                        away=non_draw[1][1],
                    )
            elif len(prices) == 2:
                # No draw (can happen in some markets) — skip
                pass

        elif key == "totals" and len(outcomes) >= 2:
            over = under = None
            line = None
            for o in outcomes:
                if o.get("name", "").lower() == "over":
                    over = o.get("price")
                    line = o.get("point", 2.5)
                elif o.get("name", "").lower() == "under":
                    under = o.get("price")
                    if line is None:
                        line = o.get("point", 2.5)
            if over and under and line is not None:
                ou_odds = OverUnderOdds(line=line, over=over, under=under)

    return odds_1x2, ou_odds


def fetch_todays_match_odds(
    api_key: Optional[str] = None,
    hours_ahead: int = 24,
) -> dict[tuple[str, str], dict]:
    """
    Fetch upcoming WC match odds from The Odds API.

    Returns dict keyed by (home_team_canonical, away_team_canonical) with:
      {"odds_1x2": MatchOdds1X2, "ou_odds": OverUnderOdds|None, "stage": TournamentStage}

    Returns empty dict on any failure (pipeline will log and skip gracefully).
    """
    api_key = api_key or os.environ.get("THE_ODDS_API_KEY", "")
    if not api_key:
        print("[odds] THE_ODDS_API_KEY not set — skipping auto-odds.")
        return {}

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)

    params = {
        "apiKey":       api_key,
        "regions":      _PREFERRED_REGIONS,
        "markets":      "h2h,totals",
        "oddsFormat":   "decimal",
    }

    try:
        resp = requests.get(_ODDS_API_ENDPOINT, params=params, timeout=20)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")
        print(f"[odds] API quota: {used} used / {remaining} remaining this month.")
        resp.raise_for_status()
        events = resp.json()
    except Exception as exc:
        print(f"[odds] Failed to fetch from The Odds API: {exc}")
        return {}

    result: dict[tuple[str, str], dict] = {}

    for event in events:
        commence = _parse_commence_time(event.get("commence_time", ""))

        # Filter to matches starting within the next `hours_ahead` hours
        if not (now <= commence <= cutoff):
            continue

        raw_home = event.get("home_team", "")
        raw_away = event.get("away_team", "")
        home_canonical = _normalize(raw_home)
        away_canonical = _normalize(raw_away)

        stage = _infer_stage(commence)

        # Try bookmakers in order until we get a valid h2h
        best_1x2:  Optional[MatchOdds1X2]  = None
        best_ou:   Optional[OverUnderOdds] = None

        for bookmaker in event.get("bookmakers", []):
            odds_1x2, ou_odds = _extract_odds_from_bookmaker(bookmaker)
            if odds_1x2 is not None:
                best_1x2 = odds_1x2
                if ou_odds is not None:
                    best_ou = ou_odds
                if best_ou is not None:
                    break  # have both markets — stop searching

        if best_1x2 is None:
            print(f"[odds] No valid h2h odds found for {raw_home} vs {raw_away} — skipping.")
            continue

        result[(home_canonical, away_canonical)] = {
            "odds_1x2": best_1x2,
            "ou_odds":  best_ou,
            "stage":    stage,
            "_raw_home": raw_home,
            "_raw_away": raw_away,
        }
        print(f"[odds] {raw_home} vs {raw_away} [{stage.value}]: "
              f"H={best_1x2.home} D={best_1x2.draw} A={best_1x2.away}"
              + (f" | O/U {best_ou.line}" if best_ou else ""))

    print(f"[odds] Total matches with odds today: {len(result)}")
    return result
