"""
context_fetcher.py — Pre-match context from API-Football (via RapidAPI).

Free tier: 100 calls/day.
Call budget per day:
  Morning run  : 1 (fixture list) + N×3 (injuries + home-form + away-form) ≤ 25 for N=8
  Lineup run   : N×1 (lineups) ≤ 8
  Grand total  : ≤ 33 calls ≪ 100 limit.

Register free at: https://rapidapi.com/api-sports/api/api-football
Add key as GitHub Secret: RAPIDAPI_KEY
FIFA World Cup 2026: league_id=1, season=2026
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import requests

_BASE_URL  = "https://api-football-v1.p.rapidapi.com/v3"
_API_HOST  = "api-football-v1.p.rapidapi.com"
_WC_LEAGUE = 1
_WC_SEASON = 2026
_TIMEOUT   = 15


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass
class MatchContext:
    home_team: str
    away_team: str
    fixture_id:   Optional[int] = None
    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None
    home_form:    str = ""          # e.g. "WWDLW" (oldest → newest)
    away_form:    str = ""
    home_injuries: list[str] = field(default_factory=list)   # e.g. ["Mbappé (knee)"]
    away_injuries: list[str] = field(default_factory=list)
    home_lineup:   list[str] = field(default_factory=list)   # confirmed starting XI
    away_lineup:   list[str] = field(default_factory=list)
    lineups_confirmed: bool = False

    @property
    def has_context(self) -> bool:
        return bool(
            self.home_form or self.away_form
            or self.home_injuries or self.away_injuries
        )

    def to_prompt_section(self) -> str:
        """Format context for the Claude prompt."""
        lines = []
        if self.home_form:
            lines.append(f"{self.home_team} recent form (oldest→newest): {self.home_form}")
        if self.away_form:
            lines.append(f"{self.away_team} recent form (oldest→newest): {self.away_form}")
        if self.home_injuries:
            lines.append(f"{self.home_team} injuries/suspensions: {', '.join(self.home_injuries)}")
        else:
            lines.append(f"{self.home_team}: no significant injury concerns reported")
        if self.away_injuries:
            lines.append(f"{self.away_team} injuries/suspensions: {', '.join(self.away_injuries)}")
        else:
            lines.append(f"{self.away_team}: no significant injury concerns reported")
        if self.lineups_confirmed and self.home_lineup:
            lines.append(f"\n{self.home_team} confirmed XI: {', '.join(self.home_lineup)}")
            lines.append(f"{self.away_team} confirmed XI: {', '.join(self.away_lineup)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _headers(api_key: str) -> dict:
    return {
        "X-RapidAPI-Key":  api_key,
        "X-RapidAPI-Host": _API_HOST,
    }


def _get(endpoint: str, params: dict, api_key: str) -> Optional[dict]:
    """GET request to API-Football; returns parsed JSON or None on any error."""
    try:
        resp = requests.get(
            f"{_BASE_URL}/{endpoint}",
            headers=_headers(api_key),
            params=params,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errors"):
            print(f"[context] API error for /{endpoint}: {data['errors']}")
            return None
        return data
    except Exception as exc:
        print(f"[context] Request failed for /{endpoint}: {exc}")
        return None


def _normalize(name: str) -> str:
    return name.lower().strip()


def _team_matches(api_name: str, schedule_name: str) -> bool:
    """True if API team name and schedule team name refer to the same team."""
    a = _normalize(api_name)
    s = _normalize(schedule_name)
    return a == s or a in s or s in a


def _find_fixture(home_team: str, away_team: str, api_key: str) -> Optional[dict]:
    """
    Fetch today's WC fixtures and return the one matching home_team/away_team.
    Tries normal order first, then swapped (WC neutral venues may list differently).
    """
    today = date.today().isoformat()
    data = _get("fixtures", {
        "league": _WC_LEAGUE,
        "season": _WC_SEASON,
        "date":   today,
    }, api_key)
    if not data:
        return None

    fixtures = data.get("response", [])
    # Normal order
    for fx in fixtures:
        teams = fx.get("teams", {})
        if (_team_matches(teams.get("home", {}).get("name", ""), home_team)
                and _team_matches(teams.get("away", {}).get("name", ""), away_team)):
            return fx
    # Swapped (neutral venues)
    for fx in fixtures:
        teams = fx.get("teams", {})
        if (_team_matches(teams.get("home", {}).get("name", ""), away_team)
                and _team_matches(teams.get("away", {}).get("name", ""), home_team)):
            print(f"[context]   NOTE: API listed teams in reversed order — home/away swapped.")
            return fx

    print(f"[context] Fixture not found for {home_team} vs {away_team} on {today}")
    return None


def _fetch_injuries(
    fixture_id: int,
    home_team_id: int,
    away_team_id: int,
    api_key: str,
) -> tuple[list[str], list[str]]:
    """Return (home_injuries, away_injuries) as human-readable strings."""
    data = _get("injuries", {"fixture": fixture_id}, api_key)
    if not data:
        return [], []

    home_inj: list[str] = []
    away_inj: list[str] = []
    for p in data.get("response", []):
        player  = p.get("player", {}).get("name", "Unknown")
        reason  = p.get("player", {}).get("reason", "")
        team_id = p.get("team", {}).get("id")
        entry   = f"{player} ({reason})" if reason else player
        if team_id == home_team_id:
            home_inj.append(entry)
        elif team_id == away_team_id:
            away_inj.append(entry)

    return home_inj, away_inj


def _fetch_team_form(team_id: int, api_key: str) -> str:
    """
    Return a form string like 'WWDLW' (last 5 WC results, oldest → newest).
    Empty string if no data.
    """
    data = _get("fixtures", {
        "team":   team_id,
        "last":   5,
        "league": _WC_LEAGUE,
        "season": _WC_SEASON,
    }, api_key)
    if not data:
        return ""

    results: list[str] = []
    for fx in data.get("response", []):
        teams = fx.get("teams", {})
        goals = fx.get("goals", {})
        if teams.get("home", {}).get("id") == team_id:
            g_team, g_opp = goals.get("home"), goals.get("away")
        else:
            g_team, g_opp = goals.get("away"), goals.get("home")

        if g_team is None or g_opp is None:
            continue
        if g_team > g_opp:
            results.append("W")
        elif g_team == g_opp:
            results.append("D")
        else:
            results.append("L")

    # API returns newest→oldest; reverse to show oldest→newest
    return "".join(reversed(results)) if results else ""


def _fetch_lineups(
    fixture_id: int,
    home_team_id: int,
    away_team_id: int,
    api_key: str,
) -> tuple[list[str], list[str], bool]:
    """Return (home_xi, away_xi, confirmed). Player name lists, empty if not announced."""
    data = _get("fixtures/lineups", {"fixture": fixture_id}, api_key)
    if not data or not data.get("response"):
        return [], [], False

    home_xi: list[str] = []
    away_xi: list[str] = []

    for team_lineup in data["response"]:
        team_id = team_lineup.get("team", {}).get("id")
        names   = [
            p.get("player", {}).get("name", "")
            for p in team_lineup.get("startXI", [])
        ]
        names = [n for n in names if n]
        if team_id == home_team_id:
            home_xi = names
        elif team_id == away_team_id:
            away_xi = names

    confirmed = bool(home_xi or away_xi)
    return home_xi, away_xi, confirmed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_match_context(
    home_team: str,
    away_team: str,
    include_lineups: bool = False,
    api_key: Optional[str] = None,
) -> Optional[MatchContext]:
    """
    Fetch pre-match context for a WC match from API-Football.

    Args:
        home_team: Schedule home team name (canonical form).
        away_team: Schedule away team name (canonical form).
        include_lineups: If True, also fetch confirmed starting XIs (pre-match run).
        api_key: RAPIDAPI_KEY. Falls back to env var.

    Returns:
        MatchContext with available data, or None if key missing / fixture not found.
        Never raises — all errors are caught and logged.
    """
    api_key = api_key or os.environ.get("RAPIDAPI_KEY", "")
    if not api_key:
        print("[context] RAPIDAPI_KEY not set — skipping pre-match context.")
        return None

    print(f"[context] Fetching context for {home_team} vs {away_team}...")

    fixture_data = _find_fixture(home_team, away_team, api_key)
    if fixture_data is None:
        return None

    fixture_id = fixture_data.get("fixture", {}).get("id")
    home_id    = fixture_data.get("teams", {}).get("home", {}).get("id")
    away_id    = fixture_data.get("teams", {}).get("away", {}).get("id")

    ctx = MatchContext(
        home_team    = home_team,
        away_team    = away_team,
        fixture_id   = fixture_id,
        home_team_id = home_id,
        away_team_id = away_id,
    )

    # Injuries
    if fixture_id and home_id and away_id:
        ctx.home_injuries, ctx.away_injuries = _fetch_injuries(
            fixture_id, home_id, away_id, api_key
        )
        print(f"[context]   {home_team} injuries: {ctx.home_injuries or 'none'}")
        print(f"[context]   {away_team} injuries: {ctx.away_injuries or 'none'}")

    # Form (last 5 WC results per team)
    if home_id:
        ctx.home_form = _fetch_team_form(home_id, api_key)
        print(f"[context]   {home_team} form: {ctx.home_form or 'N/A'}")
    if away_id:
        ctx.away_form = _fetch_team_form(away_id, api_key)
        print(f"[context]   {away_team} form: {ctx.away_form or 'N/A'}")

    # Confirmed lineups (only in --lineup-check runs)
    if include_lineups and fixture_id and home_id and away_id:
        ctx.home_lineup, ctx.away_lineup, ctx.lineups_confirmed = _fetch_lineups(
            fixture_id, home_id, away_id, api_key
        )
        if ctx.lineups_confirmed:
            print(f"[context]   Lineups confirmed!")
            print(f"[context]   {home_team} XI: {ctx.home_lineup}")
            print(f"[context]   {away_team} XI: {ctx.away_lineup}")
        else:
            print(f"[context]   Lineups not yet announced.")

    return ctx
