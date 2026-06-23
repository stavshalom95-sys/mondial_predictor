"""
scripts/sync_tables.py — Sync data/group_tables.json from live standings.

Primary source : football-data.org /v4/competitions/WC/standings
Fallback source: compute standings from the schedule JSON (uses match scores)

Usage:
    # Live fetch (requires FOOTBALL_DATA_API_KEY):
    python scripts/sync_tables.py

    # Compute from local schedule JSON (no API key needed):
    python scripts/sync_tables.py --from-schedule tests/sample_games.json

    # Custom output path:
    python scripts/sync_tables.py --output data/group_tables.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# Reuse the motivation module for status computation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.motivation import compute_group_statuses

API_BASE = "https://api.football-data.org/v4"


# ---------------------------------------------------------------------------
# Source A: football-data.org standings API
# ---------------------------------------------------------------------------

def _fetch_api_standings(api_key: str) -> dict:
    """
    GET /v4/competitions/WC/standings and return normalised group_tables dict.

    football-data.org response shape:
      {"standings": [{"type": "TOTAL", "group": "GROUP_A", "table": [...]}, ...]}

    Each table row has:
      position, team.name, playedGames, won, draw, lost,
      points, goalsFor, goalsAgainst, goalDifference
    """
    if not _REQUESTS_OK:
        raise ImportError("'requests' package not installed — cannot fetch from API.")

    headers = {"X-Auth-Token": api_key}
    resp = _requests.get(
        f"{API_BASE}/competitions/WC/standings",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    groups: dict[str, list[dict]] = {}

    for standing in data.get("standings", []):
        if standing.get("type") != "TOTAL":
            continue
        raw_group = standing.get("group", "")    # e.g. "GROUP_A"
        grp_letter = raw_group.replace("GROUP_", "").strip() or raw_group

        rows: list[dict] = []
        for row in standing.get("table", []):
            team_info = row.get("team", {})
            entry = {
                "name":             team_info.get("name", team_info.get("shortName", "")),
                "played":           int(row.get("playedGames", 0)),
                "won":              int(row.get("won", 0)),
                "drawn":            int(row.get("draw", 0)),
                "lost":             int(row.get("lost", 0)),
                "goals_for":        int(row.get("goalsFor", 0)),
                "goals_against":    int(row.get("goalsAgainst", 0)),
                "goal_difference":  int(row.get("goalDifference", 0)),
                "points":           int(row.get("points", 0)),
                "position":         int(row.get("position", 0)),
            }
            rows.append(entry)

        if rows:
            compute_group_statuses(rows)   # mutates in-place to add qualification_status
            groups[grp_letter] = rows

    return groups


# ---------------------------------------------------------------------------
# Source B: compute standings from schedule JSON results
# ---------------------------------------------------------------------------

def _compute_from_schedule(schedule_path: str) -> dict:
    """
    Parse a schedule JSON and compute group standings from finished match scores.

    Works with the football-data.org → fetch_schedule.py JSON format.
    Only 'final' status matches contribute to the standings.
    """
    with open(schedule_path, encoding="utf-8") as f:
        games = json.load(f)
    if not isinstance(games, list):
        games = games.get("games", games.get("matches", []))

    # Table is keyed by (group, team_name) but we don't have group info in
    # our minimal schedule format. Best we can do: put all teams in one group.
    # If stage info includes group hints, use them; otherwise bucket by game pairs.
    # Since football-data.org full schedule includes group field, try that first.

    team_rows: dict[str, dict] = {}  # team_name → stats dict
    group_of:  dict[str, str]  = {}  # team_name → group letter

    for g in games:
        if g.get("status") != "final":
            continue

        stage_raw = g.get("stage", "")
        # Infer group letter from stage field if available (e.g. "GROUP_A")
        grp = ""
        if "GROUP_" in stage_raw.upper():
            grp = stage_raw.upper().replace("GROUP_", "").strip()
        # Fallback: football-data.org sometimes stores it in a "group" key
        grp = grp or g.get("group", "").replace("GROUP_", "").strip() or "?"

        teams  = g.get("teams", {})
        h_key  = g.get("home", "")
        a_key  = g.get("away", "")
        h_name = teams.get(h_key, {}).get("name", h_key)
        a_name = teams.get(a_key, {}).get("name", a_key)

        score  = g.get("score", {})
        h_gf   = score.get(h_key)
        a_gf   = score.get(a_key)
        if h_gf is None or a_gf is None:
            continue
        h_gf, a_gf = int(h_gf), int(a_gf)

        # Ensure team entry exists
        for name in (h_name, a_name):
            if name not in team_rows:
                team_rows[name] = {
                    "name": name, "played": 0, "won": 0, "drawn": 0, "lost": 0,
                    "goals_for": 0, "goals_against": 0,
                    "goal_difference": 0, "points": 0,
                }

        # Track group assignment (last seen wins — fine for single group)
        group_of[h_name] = grp
        group_of[a_name] = grp

        # Update stats
        h = team_rows[h_name]
        a = team_rows[a_name]

        h["played"] += 1; a["played"] += 1
        h["goals_for"] += h_gf;      h["goals_against"] += a_gf
        a["goals_for"] += a_gf;      a["goals_against"] += h_gf
        h["goal_difference"] = h["goals_for"] - h["goals_against"]
        a["goal_difference"] = a["goals_for"] - a["goals_against"]

        if h_gf > a_gf:
            h["won"] += 1; h["points"] += 3; a["lost"] += 1
        elif h_gf < a_gf:
            a["won"] += 1; a["points"] += 3; h["lost"] += 1
        else:
            h["drawn"] += 1; h["points"] += 1
            a["drawn"] += 1; a["points"] += 1

    # Bucket teams by group letter
    bucket: dict[str, list[dict]] = {}
    for name, row in team_rows.items():
        grp = group_of.get(name, "?")
        bucket.setdefault(grp, []).append(row)

    # Sort and compute qualification status per group
    groups: dict[str, list[dict]] = {}
    for grp_letter, rows in sorted(bucket.items()):
        compute_group_statuses(rows)
        groups[grp_letter] = sorted(rows, key=lambda r: (-r["points"], -r["goal_difference"]))

    return groups


# ---------------------------------------------------------------------------
# Build output dict + write JSON
# ---------------------------------------------------------------------------

def _build_output(groups: dict, matchday: int = 0) -> dict:
    """Wrap group data in the full output structure."""
    return {
        "_note": (
            "Generated by scripts/sync_tables.py. "
            "Re-run before each daily pipeline to get live standings."
        ),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "matchday": matchday,
        "groups": groups,
    }


def _detect_matchday(groups: dict) -> int:
    """Infer current matchday from maximum played games across all teams."""
    max_played = 0
    for rows in groups.values():
        for row in rows:
            max_played = max(max_played, row.get("played", 0))
    return max_played + 1   # next matchday to be played


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync data/group_tables.json from live standings or schedule JSON."
    )
    parser.add_argument(
        "--from-schedule",
        metavar="SCHEDULE_JSON",
        default=None,
        help="Compute standings from a local schedule JSON instead of calling the API.",
    )
    parser.add_argument(
        "--output",
        default="data/group_tables.json",
        help="Output path (default: data/group_tables.json)",
    )
    args = parser.parse_args()

    if args.from_schedule:
        print(f"[sync_tables] Computing standings from '{args.from_schedule}'...")
        groups = _compute_from_schedule(args.from_schedule)
        source = f"computed from {args.from_schedule}"
    else:
        api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
        if not api_key:
            print(
                "[sync_tables] FOOTBALL_DATA_API_KEY not set — "
                "use --from-schedule to compute from local JSON.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[sync_tables] Fetching WC standings from {API_BASE}...")
        groups = _fetch_api_standings(api_key)
        source = "football-data.org API"

    if not groups:
        print("[sync_tables] No group data found. Nothing written.", file=sys.stderr)
        sys.exit(1)

    matchday = _detect_matchday(groups)
    output   = _build_output(groups, matchday)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_teams = sum(len(rows) for rows in groups.values())
    print(
        f"[sync_tables] Written {len(groups)} group(s), {total_teams} team(s) "
        f"(matchday {matchday}) -> '{args.output}'  [source: {source}]"
    )

    # Print qualification status summary
    for grp, rows in sorted(groups.items()):
        print(f"  Group {grp}:")
        for row in rows:
            print(
                f"    {row['name']:<22} {row['points']}pts  "
                f"GD={row['goal_difference']:+d}  -> {row.get('qualification_status','?')}"
            )


if __name__ == "__main__":
    main()
