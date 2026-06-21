#!/usr/bin/env python3
"""
Fetch the World Cup 2026 schedule from football-data.org and write it as JSON.

Used by GitHub Actions before running main.py.
Output format is fully compatible with data_pipeline.parse_world_cup_schedule().

Usage:
  python scripts/fetch_schedule.py --output tests/sample_games.json

Requires env var:
  FOOTBALL_DATA_API_KEY  (free registration at https://www.football-data.org/client/register)
"""
import argparse
import json
import os
import sys

import requests

API_BASE = "https://api.football-data.org/v4"

# football-data.org status -> our internal status
STATUS_MAP: dict[str, str] = {
    "FINISHED":       "final",
    "IN_PLAY":        "live",
    "PAUSED":         "live",
    "HALFTIME":       "live",
    "EXTRA_TIME":     "live",
    "PENALTY":        "live",
    "SCHEDULED":      "scheduled",
    "TIMED":          "scheduled",
    "POSTPONED":      "scheduled",
    "SUSPENDED":      "scheduled",
    "CANCELLED":      "scheduled",
}

# football-data.org stage -> our internal stage key (used by data_pipeline for stage inference)
STAGE_MAP: dict[str, str] = {
    "GROUP_STAGE":    "group_stage",
    "LAST_32":        "round_of_32",
    "LAST_16":        "round_of_16",
    "QUARTER_FINALS": "quarter_final",
    "SEMI_FINALS":    "semi_final",
    "THIRD_PLACE":    "third_place",
    "FINAL":          "final_stage",
}


def fetch_wc_matches(api_key: str) -> list[dict]:
    headers = {"X-Auth-Token": api_key}
    resp = requests.get(
        f"{API_BASE}/competitions/WC/matches",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    matches: list[dict] = []
    for m in data.get("matches", []):
        home_info = m.get("homeTeam", {})
        away_info = m.get("awayTeam", {})

        # Prefer TLA (3-letter abbreviation), fall back to shortName
        home_abbr = home_info.get("tla") or home_info.get("shortName", "HOM")
        away_abbr = away_info.get("tla") or away_info.get("shortName", "AWY")

        home_name = home_info.get("name", home_abbr)
        away_name = away_info.get("name", away_abbr)

        score_full = m.get("score", {}).get("fullTime", {})
        home_score = score_full.get("home")   # None if not played yet
        away_score = score_full.get("away")

        raw_status = m.get("status", "SCHEDULED")
        status = STATUS_MAP.get(raw_status, "scheduled")

        raw_stage = m.get("stage", "GROUP_STAGE")
        stage = STAGE_MAP.get(raw_stage, "group_stage")

        matches.append({
            "id":         str(m.get("id", "")),
            "status":     status,
            "stage":      stage,
            "start_time": m.get("utcDate", ""),
            "home":       home_abbr,
            "away":       away_abbr,
            "teams": {
                home_abbr: {"name": home_name, "abbreviation": home_abbr},
                away_abbr: {"name": away_name, "abbreviation": away_abbr},
            },
            "score": {
                home_abbr: home_score,
                away_abbr: away_score,
            },
        })

    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch WC 2026 schedule from football-data.org")
    parser.add_argument(
        "--output",
        default="tests/sample_games.json",
        help="Path to write the JSON output (default: tests/sample_games.json)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not api_key:
        print("ERROR: FOOTBALL_DATA_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching WC 2026 schedule from {API_BASE}...")
    matches = fetch_wc_matches(api_key)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(matches)} matches to {args.output}")


if __name__ == "__main__":
    main()
