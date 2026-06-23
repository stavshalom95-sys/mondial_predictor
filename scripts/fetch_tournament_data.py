"""
scripts/fetch_tournament_data.py — Fetch live 2026 World Cup data from API-Football.

Fetches:
  1. Group standings  (/standings)
  2. Remaining fixtures — status NS (not started)  (/fixtures)
  3. Top scorers  (/players/topscorers)

Saves to tournament_data.json in the project root and prints a brief summary.

Usage:
    python scripts/fetch_tournament_data.py

Requires RAPIDAPI_KEY in environment (same key used by context_fetcher.py).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from data.context_fetcher import _get, _WC_LEAGUE, _WC_SEASON

_OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "tournament_data.json")


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_standings(api_key: str) -> list[dict]:
    """
    Return a list of group dicts, each containing the group name and ranked teams.

    Shape:
      [
        {
          "group": "Group A",
          "standings": [
            {"rank": 1, "team": "...", "played": 3, "won": 2, "drawn": 1, "lost": 0,
             "goals_for": 5, "goals_against": 2, "goal_diff": 3, "points": 7, "form": "WWD"},
            ...
          ]
        },
        ...
      ]
    """
    print("[fetch] Requesting standings...")
    data = _get("standings", {"league": _WC_LEAGUE, "season": _WC_SEASON}, api_key)
    if not data:
        print("[fetch] WARNING: standings returned no data.")
        return []

    groups = []
    # Response structure: data["response"][0]["league"]["standings"] is a list of groups
    try:
        raw_groups = data["response"][0]["league"]["standings"]
    except (IndexError, KeyError) as exc:
        print(f"[fetch] WARNING: unexpected standings shape: {exc}")
        return []

    for raw_group in raw_groups:
        # Each raw_group is a list of team-standing dicts
        if not raw_group:
            continue
        group_name = raw_group[0].get("group", "Unknown Group")
        teams = []
        for entry in raw_group:
            all_stats = entry.get("all", {})
            goals     = all_stats.get("goals", {})
            teams.append({
                "rank":          entry.get("rank"),
                "team":          entry.get("team", {}).get("name", ""),
                "team_id":       entry.get("team", {}).get("id"),
                "played":        all_stats.get("played", 0),
                "won":           all_stats.get("win",    0),
                "drawn":         all_stats.get("draw",   0),
                "lost":          all_stats.get("lose",   0),
                "goals_for":     goals.get("for",     0),
                "goals_against": goals.get("against", 0),
                "goal_diff":     entry.get("goalsDiff", 0),
                "points":        entry.get("points", 0),
                "form":          entry.get("form", ""),
            })
        groups.append({"group": group_name, "standings": teams})

    print(f"[fetch] Standings: {len(groups)} group(s) received.")
    return groups


def fetch_remaining_fixtures(api_key: str) -> list[dict]:
    """
    Return fixtures with status NS (Not Started) — i.e., the remaining schedule.

    Shape:
      [
        {
          "fixture_id": 123,
          "round": "Round of 16",
          "date": "2026-07-04T19:00:00+00:00",
          "venue": "SoFi Stadium",
          "home_team": "...",
          "home_team_id": ...,
          "away_team": "...",
          "away_team_id": ...,
          "status": "NS",
        },
        ...
      ]
    """
    print("[fetch] Requesting remaining fixtures (status=NS)...")
    data = _get("fixtures", {
        "league":  _WC_LEAGUE,
        "season":  _WC_SEASON,
        "status":  "NS",   # Not Started
    }, api_key)

    if not data:
        print("[fetch] WARNING: fixtures returned no data.")
        return []

    fixtures = []
    for fx in data.get("response", []):
        fix_info = fx.get("fixture", {})
        teams    = fx.get("teams", {})
        league   = fx.get("league", {})
        fixtures.append({
            "fixture_id":   fix_info.get("id"),
            "round":        league.get("round", ""),
            "date":         fix_info.get("date", ""),
            "venue":        fix_info.get("venue", {}).get("name", ""),
            "city":         fix_info.get("venue", {}).get("city", ""),
            "home_team":    teams.get("home", {}).get("name", ""),
            "home_team_id": teams.get("home", {}).get("id"),
            "away_team":    teams.get("away", {}).get("name", ""),
            "away_team_id": teams.get("away", {}).get("id"),
            "status":       fix_info.get("status", {}).get("short", "NS"),
        })

    # Sort chronologically
    fixtures.sort(key=lambda x: x.get("date", ""))
    print(f"[fetch] Remaining fixtures: {len(fixtures)} match(es) returned.")
    return fixtures


def fetch_top_scorers(api_key: str, limit: int = 20) -> list[dict]:
    """
    Return top scorers for the WC 2026.

    Shape:
      [
        {
          "rank": 1,
          "player": "Kylian Mbappe",
          "player_id": ...,
          "nationality": "France",
          "team": "France",
          "team_id": ...,
          "goals": 5,
          "assists": 2,
          "penalties": 1,
          "appearances": 4,
          "minutes": 360,
        },
        ...
      ]
    """
    print("[fetch] Requesting top scorers...")
    data = _get("players/topscorers", {
        "league": _WC_LEAGUE,
        "season": _WC_SEASON,
    }, api_key)

    if not data:
        print("[fetch] WARNING: topscorers returned no data.")
        return []

    scorers = []
    for rank, entry in enumerate(data.get("response", [])[:limit], start=1):
        player   = entry.get("player", {})
        stats    = entry.get("statistics", [{}])[0]
        goals    = stats.get("goals", {})
        games    = stats.get("games", {})
        penalty  = stats.get("penalty", {})
        team     = stats.get("team", {})
        scorers.append({
            "rank":        rank,
            "player":      player.get("name", ""),
            "player_id":   player.get("id"),
            "nationality": player.get("nationality", ""),
            "team":        team.get("name", ""),
            "team_id":     team.get("id"),
            "goals":       goals.get("total", 0) or 0,
            "assists":     goals.get("assists", 0) or 0,
            "penalties":   penalty.get("scored", 0) or 0,
            "appearances": games.get("appearences", 0) or 0,
            "minutes":     games.get("minutes", 0) or 0,
        })

    print(f"[fetch] Top scorers: {len(scorers)} player(s) returned.")
    return scorers


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(standings: list[dict], scorers: list[dict], fixtures: list[dict]) -> None:
    print()
    print("=" * 68)
    print("  2026 WORLD CUP — TOURNAMENT DATA SNAPSHOT")
    print(f"  Fetched: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 68)

    # Group leaders
    print("\n  GROUP LEADERS")
    print("  " + "-" * 60)
    for g in standings:
        leader = g["standings"][0] if g["standings"] else None
        if not leader:
            continue
        form_str = f"  form={leader['form']}" if leader.get("form") else ""
        print(
            f"  {g['group']:<14}  {leader['team']:<22} "
            f"{leader['points']}pts  "
            f"W{leader['won']} D{leader['drawn']} L{leader['lost']}  "
            f"GD{leader['goal_diff']:+d}"
            f"{form_str}"
        )

    # Top scorers
    print("\n  TOP SCORERS")
    print("  " + "-" * 60)
    for s in scorers[:10]:
        pen_note = f"  ({s['penalties']} pen)" if s["penalties"] else ""
        print(
            f"  {s['rank']:>2}. {s['player']:<24} {s['team']:<20} "
            f"{s['goals']} goal(s){pen_note}"
        )

    # Remaining fixtures (next 10)
    print(f"\n  REMAINING FIXTURES  ({len(fixtures)} total)")
    print("  " + "-" * 60)
    for fx in fixtures[:10]:
        try:
            dt = datetime.fromisoformat(fx["date"].replace("Z", "+00:00"))
            date_str = dt.strftime("%d %b %H:%M UTC")
        except Exception:
            date_str = fx.get("date", "TBD")[:16]
        print(
            f"  {date_str}  {fx['round']:<22}"
            f"  {fx['home_team']} vs {fx['away_team']}"
        )
    if len(fixtures) > 10:
        print(f"  ... and {len(fixtures) - 10} more fixtures")

    print("=" * 68)
    print(f"  Saved to: tournament_data.json")
    print("=" * 68)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("RAPIDAPI_KEY", "")
    if not api_key:
        print("[fetch] ERROR: RAPIDAPI_KEY not set in environment.")
        print("        Export it: set RAPIDAPI_KEY=your_key   (Windows)")
        print("                   export RAPIDAPI_KEY=your_key (Unix)")
        sys.exit(1)

    standings = fetch_standings(api_key)
    fixtures  = fetch_remaining_fixtures(api_key)
    scorers   = fetch_top_scorers(api_key)

    payload = {
        "fetched_at":          datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "league_id":           _WC_LEAGUE,
        "season":              _WC_SEASON,
        "group_standings":     standings,
        "remaining_fixtures":  fixtures,
        "top_scorers":         scorers,
    }

    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print_summary(standings, scorers, fixtures)


if __name__ == "__main__":
    main()
