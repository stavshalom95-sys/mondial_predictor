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


def _candidate_matches(
    raw_games: list[dict],
    now: datetime,
) -> list[tuple[str, str, datetime, str]]:
    """
    Return ALL matches within a ±48-hour window as
    (home_name, away_name, start_time, status) tuples.

    The wider window means the main sync() loop receives EVERY candidate and
    can apply the explicit 3-case decision tree:
        CASE 1 — past/finished  → remove stale entry (never preserve)
        CASE 2 — upcoming/live  → preserve if odds > 0, else blank
        CASE 3 — new fixture    → add blank placeholder

    Using the broader window instead of a hard future-only filter ensures
    the logic is visible in the calling code, not hidden in this function.
    """
    cutoff_future = now + timedelta(hours=48)
    cutoff_past   = now - timedelta(hours=24)   # discard very old fixtures

    result: list[tuple[str, str, datetime, str]] = []

    for g in raw_games:
        start_raw = g.get("start_time", "")
        try:
            start_time = datetime.fromisoformat(start_raw)
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        # Outside the window entirely — skip (yesterday's early games, far future)
        if not (cutoff_past <= start_time <= cutoff_future):
            continue

        teams     = g.get("teams", {})
        home_key  = g.get("home", "")
        away_key  = g.get("away", "")
        home_name = teams.get(home_key, {}).get("name", home_key)
        away_name = teams.get(away_key, {}).get("name", away_key)
        status    = g.get("status", "scheduled")

        result.append((home_name, away_name, start_time, status))

    return result


def _blank_entry() -> dict:
    """Placeholder entry for a new match — user fills in odds."""
    return {
        "winner":          {"home": 0.0, "draw": 0.0, "away": 0.0},
        "sum_goals":       {"0-1": 0.0, "2-3": 0.0, "+4": 0.0},
        "corners_range":   {"0-8": 0.0, "9-11": 0.0, "12+": 0.0},
    }


def _has_nonzero_odds(entry: dict) -> bool:
    """Return True if ANY numeric odds value in the entry is > 0.0."""
    for sub in entry.values():
        if isinstance(sub, dict):
            if any(isinstance(v, (int, float)) and v > 0.0 for v in sub.values()):
                return True
    return False


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

    now = datetime.now(timezone.utc)
    candidates = _candidate_matches(raw_games, now)

    if not candidates:
        print("[sync] No matches found in the ±48h window. winner_odds.json not modified.")
        return

    existing = _load_existing(odds_path)

    new_data: dict = {}
    new_data["_note"] = (
        f"Synced by sync_winner_odds.py on {now.strftime('%Y-%m-%dT%H:%M:%SZ')}. "
        "Filled odds are preserved; blanks need to be filled before the pipeline runs."
    )

    preserved = 0
    added     = 0
    zeroed    = 0
    stale     = 0

    for home, away, start_time, status in candidates:
        if not home or not away:
            continue   # KO bracket TBD — skip

        canonical_key = _key(home, away)

        # ── Look up any existing entry for this match ─────────────────────
        existing_entry = None
        for k, v in existing.items():
            if k.startswith("_"):
                continue
            if _keys_match(k, home, away):
                existing_entry = v
                break

        # ── CASE 1: match already played / in the past ────────────────────
        # Condition: status=="final"  OR  kick-off time has already passed.
        # Do NOT preserve odds for finished games — they are stale data.
        # The entry is simply omitted from new_data (effectively removed).
        is_past = (status == "final") or (start_time <= now)
        if is_past:
            stale += 1
            label = "final" if status == "final" else f"started {start_time.strftime('%H:%M UTC')}"
            print(f"[sync] STALE      {canonical_key}  ({label} — removed from file)")
            continue   # skip: do not write this entry to new_data

        # ── CASE 2: upcoming match with filled odds ───────────────────────
        # Preserve regardless of --reset; real odds must never be wiped.
        if existing_entry is not None and _has_nonzero_odds(existing_entry):
            new_data[canonical_key] = existing_entry
            preserved += 1
            print(f"[sync] PRESERVED  {canonical_key}  (odds filled — not overwritten)")
            continue

        # ── CASE 3: upcoming match with no filled odds ────────────────────
        # --reset: explicitly zero out (idempotent since it's already 0.0).
        # no flag: write blank placeholder so the user knows to fill it.
        new_data[canonical_key] = _blank_entry()
        if reset:
            zeroed += 1
            print(f"[sync] RESET      {canonical_key}  (odds 0.0 — fill in before run!)")
        else:
            added += 1
            print(f"[sync] ADDED      {canonical_key}  (new placeholder — fill in before run!)")

    print(
        f"\n[sync] Summary: {preserved} preserved, {added} added, "
        f"{zeroed} zeroed, {stale} stale removed."
    )

    with open(odds_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    print(f"[sync] Written to '{odds_path}'.")


if __name__ == "__main__":
    schedule  = sys.argv[1] if len(sys.argv) > 1 else "tests/sample_games.json"
    output    = sys.argv[2] if len(sys.argv) > 2 else "winner_odds.json"
    force_reset = "--reset" in sys.argv
    sync(schedule, output, reset=force_reset)
