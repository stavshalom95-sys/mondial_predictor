"""
scripts/sync_winner_odds.py — Sync winner_odds.json to today's schedule.

Reads a World Cup schedule JSON, finds matches that start today (UTC),
and writes/updates winner_odds.json with the multi-market dict format.

  • Existing entries whose key still matches a today's match are PRESERVED.
  • New matches get placeholder odds of 0 (fill them in manually or via API).
  • Stale entries (yesterday's teams) are REMOVED.

Usage:
    python scripts/sync_winner_odds.py
    python scripts/sync_winner_odds.py tests/sample_games.json
    python scripts/sync_winner_odds.py tests/sample_games.json winner_odds.json
    python scripts/sync_winner_odds.py tests/sample_games.json winner_odds.json --reset

Flags:
    --reset   Zero out all odds for today's matches (daily fresh start).
              Without this flag, existing odds are preserved.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Strip emoji flag prefix, lowercase."""
    name = name.strip()
    if name and not name[0].isascii():
        parts = name.split(None, 1)
        name = parts[1] if len(parts) > 1 else ""
    return name.lower().strip()


def _key(home: str, away: str) -> str:
    return f"{home} vs {away}"


def _parse_schedule(schedule_path: str) -> list[dict]:
    """Parse the schedule JSON and return raw game dicts."""
    with open(schedule_path, encoding="utf-8") as f:
        raw = json.load(f)
    # Support both a plain list and {"games": [...]} wrapper
    if isinstance(raw, list):
        return raw
    return raw.get("games", raw.get("matches", []))


def _todays_matches(raw_games: list[dict]) -> list[tuple[str, str]]:
    """
    Return (home_name, away_name) pairs for upcoming unplayed matches.

    Rules:
      • Skip matches already kicked off (start_time <= now) — status field
        is unreliable since fetch_schedule.py only updates it once per day.
      • Include matches starting within the next 48 hours, so tomorrow's
        fixtures appear when today's games have already kicked off.
      • Also skip status == "final" as a belt-and-suspenders guard.
    """
    now    = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=48)

    result: list[tuple[str, str]] = []

    for g in raw_games:
        status = g.get("status", "scheduled")
        if status == "final":
            continue

        start_raw = g.get("start_time", "")
        try:
            start_time = datetime.fromisoformat(start_raw)
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        # Only upcoming matches: kick-off must be strictly in the future
        if not (now < start_time <= cutoff):
            continue

        teams    = g.get("teams", {})
        home_key = g.get("home", "")
        away_key = g.get("away", "")
        home_name = teams.get(home_key, {}).get("name", home_key)
        away_name = teams.get(away_key, {}).get("name", away_key)
        result.append((home_name, away_name))

    return result


def _blank_entry() -> dict:
    """Placeholder entry for a new match — user fills in odds."""
    return {
        "winner":          {"home": 0.0, "draw": 0.0, "away": 0.0},
        "sum_goals":       {"0-1": 0.0, "2-3": 0.0, "+4": 0.0},
        "corners_range":   {"0-8": 0.0, "9-11": 0.0, "12+": 0.0},
    }


def _load_existing(odds_path: str) -> dict:
    """Load existing winner_odds.json (new dict format). Returns {} if absent."""
    try:
        with open(odds_path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return raw
        # Old list format — discard; we're migrating
        print(f"[sync] Old list format detected in '{odds_path}' — starting fresh.")
        return {}
    except FileNotFoundError:
        return {}


def _keys_match(existing_key: str, home: str, away: str) -> bool:
    """Check if an existing dict key matches a given home/away pair."""
    if " vs " not in existing_key:
        return False
    k_home, k_away = existing_key.split(" vs ", 1)
    p = _norm
    return (p(k_home) == p(home) and p(k_away) == p(away)) or \
           (p(k_home) == p(away) and p(k_away) == p(home))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sync(schedule_path: str = "tests/sample_games.json",
         odds_path: str     = "winner_odds.json",
         reset: bool        = False) -> None:

    print(f"[sync] Reading schedule from '{schedule_path}'...")
    raw_games = _parse_schedule(schedule_path)
    today_pairs = _todays_matches(raw_games)

    if not today_pairs:
        print("[sync] No unfinished matches found for today (UTC). winner_odds.json not modified.")
        return

    print(f"[sync] Today's matches ({len(today_pairs)}):")
    for h, a in today_pairs:
        print(f"  {h} vs {a}")

    existing = _load_existing(odds_path)

    # Build new dict: blank all entries (reset mode) or preserve matching ones
    new_data: dict = {}

    from datetime import datetime, timezone as _tz
    new_data["_note"] = (
        f"Reset {'(forced blank)' if reset else '(preserved)'} "
        f"by sync_winner_odds.py on {datetime.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}. "
        "Fill in decimal odds before the daily pipeline runs."
    )

    preserved = 0
    added     = 0
    zeroed    = 0

    for home, away in today_pairs:
        canonical_key = _key(home, away)

        if reset:
            new_data[canonical_key] = _blank_entry()
            zeroed += 1
            print(f"[sync] RESET      {canonical_key}  (odds zeroed — fill in before run!)")
        else:
            # Try to find an existing entry that matches this match
            existing_entry = None
            for k, v in existing.items():
                if k.startswith("_"):
                    continue
                if _keys_match(k, home, away):
                    existing_entry = v
                    break

            if existing_entry is not None:
                new_data[canonical_key] = existing_entry
                preserved += 1
                print(f"[sync] PRESERVED  {canonical_key}")
            else:
                new_data[canonical_key] = _blank_entry()
                added += 1
                print(f"[sync] ADDED      {canonical_key}  (placeholder odds — fill in before run!)")

    removed = len([k for k in existing if not k.startswith("_")]) - (preserved + zeroed)
    if reset:
        print(f"\n[sync] Summary: {zeroed} zeroed, {removed} stale removed.")
    else:
        print(f"\n[sync] Summary: {preserved} preserved, {added} added, {removed} stale removed.")

    with open(odds_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    print(f"[sync] Written to '{odds_path}'.")


if __name__ == "__main__":
    schedule  = sys.argv[1] if len(sys.argv) > 1 else "tests/sample_games.json"
    output    = sys.argv[2] if len(sys.argv) > 2 else "winner_odds.json"
    force_reset = "--reset" in sys.argv
    sync(schedule, output, reset=force_reset)
