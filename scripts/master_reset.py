#!/usr/bin/env python3
"""
scripts/master_reset.py — Master Reset: sync winner_odds.json to today's live schedule.

Fixes the "pipeline stuck on yesterday's data" problem in one shot.

What it does:
  1. Fetches today's live schedule (FOOTBALL_DATA_API_KEY → football-data.org,
     or falls back to tests/sample_games.json when the key is absent).
  2. Parses today's unfinished matches using the same midnight-UTC window as main.py.
  3. Clears and rewrites winner_odds.json with today's match keys and EMPTY odds
     (0.0 placeholders), ready for you to fill in.
  4. Audits main.py for all critical integration points to confirm the pipeline
     is not broken.
  5. Prints a clear PASS / FAIL / ACTION-NEEDED summary.

Usage:
    python scripts/master_reset.py
    python scripts/master_reset.py --schedule tests/sample_games.json   # force local file
    python scripts/master_reset.py --output  winner_odds.json           # custom output path
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # safe on Windows cp1255

# ── Paths ─────────────────────────────────────────────────────────────────────
_DEFAULT_SCHEDULE = _ROOT / "tests" / "sample_games.json"
_DEFAULT_OUTPUT   = _ROOT / "winner_odds.json"
_MAIN_PY          = _ROOT / "main.py"

API_BASE = "https://api.football-data.org/v4"

_DIVIDER = "─" * 60


def _header(title: str) -> None:
    print(f"\n{_DIVIDER}")
    print(f"  {title}")
    print(_DIVIDER)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Fetch schedule
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_live_schedule(api_key: str) -> list[dict]:
    """Call football-data.org /v4/competitions/WC/matches and return raw game list."""
    try:
        import requests
    except ImportError:
        raise RuntimeError("'requests' not installed — pip install requests")

    STATUS_MAP = {
        "FINISHED": "final", "IN_PLAY": "live", "PAUSED": "live",
        "HALFTIME": "live",  "EXTRA_TIME": "live", "PENALTY": "live",
        "SCHEDULED": "scheduled", "TIMED": "scheduled",
        "POSTPONED": "scheduled", "SUSPENDED": "scheduled", "CANCELLED": "scheduled",
    }
    STAGE_MAP = {
        "GROUP_STAGE": "group_stage", "LAST_32": "round_of_32",
        "LAST_16": "round_of_16", "QUARTER_FINALS": "quarter_final",
        "SEMI_FINALS": "semi_final", "THIRD_PLACE": "third_place", "FINAL": "final_stage",
    }

    _now    = datetime.now(timezone.utc)
    _d_from = (_now - timedelta(days=1)).strftime("%Y-%m-%d")
    _d_to   = (_now + timedelta(days=2)).strftime("%Y-%m-%d")   # +2 to include next-day UTC slots
    resp = requests.get(
        f"{API_BASE}/competitions/WC/matches",
        headers={"X-Auth-Token": api_key},
        params={"dateFrom": _d_from, "dateTo": _d_to},
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()

    games: list[dict] = []
    for m in raw.get("matches", []):
        home = m.get("homeTeam", {})
        away = m.get("awayTeam", {})
        h_abbr = home.get("tla") or home.get("shortName", "HOM")
        a_abbr = away.get("tla") or away.get("shortName", "AWY")
        score  = m.get("score", {}).get("fullTime", {})
        games.append({
            "id":         str(m.get("id", "")),
            "status":     STATUS_MAP.get(m.get("status", "SCHEDULED"), "scheduled"),
            "stage":      STAGE_MAP.get(m.get("stage", "GROUP_STAGE"), "group_stage"),
            "start_time": m.get("utcDate", ""),
            "home": h_abbr,
            "away": a_abbr,
            "teams": {
                h_abbr: {"name": home.get("name", h_abbr), "abbreviation": h_abbr},
                a_abbr: {"name": away.get("name", a_abbr), "abbreviation": a_abbr},
            },
            "score": {h_abbr: score.get("home"), a_abbr: score.get("away")},
        })
    return games


def _load_local_schedule(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, list) else raw.get("games", raw.get("matches", []))


def fetch_schedule(force_local: Path | None = None) -> tuple[list[dict], str]:
    """Return (raw_games, source_label)."""
    if force_local:
        games = _load_local_schedule(force_local)
        return games, f"local file: {force_local}"

    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if api_key:
        try:
            games = _fetch_live_schedule(api_key)
            return games, "football-data.org API (live)"
        except Exception as exc:
            print(f"  [warn] API fetch failed: {exc}  — falling back to local file.")

    if _DEFAULT_SCHEDULE.exists():
        games = _load_local_schedule(_DEFAULT_SCHEDULE)
        return games, f"local fallback: {_DEFAULT_SCHEDULE}"

    raise FileNotFoundError(
        f"No schedule source available: set FOOTBALL_DATA_API_KEY or "
        f"ensure {_DEFAULT_SCHEDULE} exists."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Identify today's unfinished matches (same logic as get_todays_matches)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_start(raw: str) -> datetime:
    try:
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def todays_matches(raw_games: list[dict]) -> list[dict]:
    """
    Return all matches scheduled for today's UTC date or tomorrow's UTC date.

    Captures all WC 2026 slots regardless of time zone: a match at 02:00 UTC on the
    next calendar day (e.g. Colombia vs Congo DR) is included because its UTC date
    equals tomorrow.  get_todays_matches() in data_pipeline.py uses the same logic.

    Finished matches are passed through; reset_winner_odds() routes them to
    _played_today rather than the main entries section.
    """
    now_utc       = datetime.now(timezone.utc)
    today_date    = now_utc.date()
    tomorrow_date = today_date + timedelta(days=1)

    result: list[dict] = []
    for g in raw_games:
        start = _parse_start(g.get("start_time", ""))
        if start.date() in (today_date, tomorrow_date):
            # Include ALL matches (incl. finished) — finished early-morning games
            # are recorded in winner_odds.json under _played_today for traceability.
            # The main pipeline (get_todays_matches) still skips status="final" games.
            result.append(g)

    result.sort(key=lambda g: _parse_start(g.get("start_time", "")))
    return result


def _team_name(game: dict, key: str) -> str:
    return game.get("teams", {}).get(key, {}).get("name", key)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Write fresh winner_odds.json
# ─────────────────────────────────────────────────────────────────────────────

def reset_winner_odds(
    matches: list[dict], output: Path
) -> tuple[list[str], list[str]]:
    """
    Build and write winner_odds.json, preserving any entry whose odds are
    already filled in (at least one value > 0.0 in any sub-market).

    Upcoming / live matches:
      - If a matching entry exists with non-zero odds → preserve it unchanged.
      - Otherwise → write 0.0 placeholders.
    Already-finished matches → logged under _played_today (no odds entry).

    Returns (upcoming_keys, preserved_keys):
      upcoming_keys  — all non-final match keys written (preserved + new)
      preserved_keys — subset whose existing odds were kept intact
    """
    # Load existing file so we can avoid overwriting filled odds
    existing: dict = {}
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except Exception:
            pass

    def _has_nonzero_odds(entry: dict) -> bool:
        """Return True if any nested numeric odds value is > 0.0."""
        for sub in entry.values():
            if isinstance(sub, dict):
                if any(isinstance(v, (int, float)) and v > 0.0 for v in sub.values()):
                    return True
        return False

    def _find_existing_entry(home: str, away: str) -> dict | None:
        """Case-insensitive lookup for an existing odds entry by team names."""
        h, a = home.lower(), away.lower()
        for k, v in existing.items():
            if k.startswith("_") or " vs " not in k:
                continue
            k_h, k_a = (p.strip().lower() for p in k.split(" vs ", 1))
            if (k_h == h and k_a == a) or (k_h == a and k_a == h):
                if isinstance(v, dict):
                    return v
        return None

    upcoming_keys:  list[str] = []
    preserved_keys: list[str] = []
    played_keys:    list[str] = []
    data: dict = {
        "_note": (
            f"Reset by master_reset.py on {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}. "
            "Fill in decimal odds before the daily pipeline runs."
        ),
    }

    for g in matches:
        h_key     = g.get("home", "")
        a_key     = g.get("away", "")
        h_name    = _team_name(g, h_key)
        a_name    = _team_name(g, a_key)
        match_key = f"{h_name} vs {a_name}"

        if g.get("status") == "final":
            # Game already finished — log for traceability, skip odds entry
            played_keys.append(match_key)
        else:
            existing_entry = _find_existing_entry(h_name, a_name)
            if existing_entry is not None and _has_nonzero_odds(existing_entry):
                data[match_key] = existing_entry   # preserve filled odds
                preserved_keys.append(match_key)
            else:
                data[match_key] = {
                    "winner":        {"home": 0.0, "draw": 0.0, "away": 0.0},
                    "sum_goals":     {"0-1": 0.0, "2-3": 0.0, "+4": 0.0},
                    "corners_range": {"0-8": 0.0, "9-11": 0.0, "12+": 0.0},
                }
            upcoming_keys.append(match_key)

    if played_keys:
        data["_played_today"] = played_keys   # visible in the file for traceability

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return upcoming_keys, preserved_keys


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Audit main.py
# ─────────────────────────────────────────────────────────────────────────────

_AUDIT_CHECKS: list[tuple[str, str, str]] = [
    # (label, regex_pattern, fix_hint)
    (
        "calculate_all_markets import",
        r"from core\.market_calculator import calculate_all_markets",
        "Add: from core.market_calculator import calculate_all_markets",
    ),
    (
        "calculate_all_markets() call in loop",
        r"markets\s*=\s*calculate_all_markets\(",
        "Add market calc block inside the match loop (see market_calculator.py docs)",
    ),
    (
        "calibrate_dc() used (not calibrate)",
        r"model\s*=\s*calibrate_dc\(",
        "Change calibrate() -> calibrate_dc() at the model-build step",
    ),
    (
        "get_todays_matches() present",
        r"todays_matches\s*=\s*get_todays_matches\(",
        "Import and call get_todays_matches from data.data_pipeline",
    ),
    (
        "date-based match window",
        r"today_date\s*=\s*\w+\.date\(\)",
        "data_pipeline.get_todays_matches() must use today_date=now_utc.date() (date-based window)",
    ),
    (
        "motivation import",
        r"from data\.motivation import",
        "Add: from data.motivation import load_group_tables, build_match_motivation",
    ),
    (
        "build_match_motivation() in loop",
        r"build_match_motivation\(",
        "Add motivation block after strength-model blending",
    ),
    (
        "lambda_multiplier applied",
        r"lam_h\s*=\s*round\(lam_h\s*\*\s*_match_motivation\.home\.lambda_multiplier",
        "Apply motivation multiplier to lam_h and lam_a before simulate()",
    ),
    (
        "winner_odds cache pre-loaded",
        r"_winner_odds_cache\s*=\s*get_all_odds\(",
        "Call get_all_odds() before the match loop",
    ),
    (
        "sg_value_bet detection",
        r"sg_value_bet\s*=\s*_bracket|sg_value_bet.*=.*['\"]0-1['\"]|_SG_KEY_MAP",
        "Add sum-goals EV detection block after market calc (sg_value_bet = _bracket)",
    ),
    (
        "tournament_context_section passed to enhance()",
        r"tournament_context_section\s*=\s*_match_motivation\.to_ai_section\(",
        "Pass tournament_context_section= to enhance() call",
    ),
    (
        "model_sg_01 in morning_data",
        r"['\"]model_sg_01['\"]",
        "Add model_sg_01/sg_23/sg_4plus keys to morning_data.append({...})",
    ),
    (
        "enrich_picks() called after loop",
        r"morning_data\s*=\s*enrich_picks\(morning_data",
        "Call enrich_picks(morning_data, ...) after the match loop",
    ),
    (
        "tournament_context_lines in DailyPick",
        r"tournament_context_lines.*_match_motivation\.to_whatsapp_lines\(",
        "Add tournament_context_lines= to DailyPick(...) call",
    ),
]


def audit_main(main_py: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Returns (passed_labels, [(failed_label, fix_hint)])."""
    src_main     = main_py.read_text(encoding="utf-8")
    pipeline_py  = main_py.parent / "data" / "data_pipeline.py"
    src_pipeline = pipeline_py.read_text(encoding="utf-8") if pipeline_py.exists() else ""

    passed: list[str]              = []
    failed: list[tuple[str, str]]  = []

    # Most checks look in main.py; the date-based window check lives in data_pipeline.py
    _PIPELINE_ONLY = {"date-based match window"}

    for label, pattern, hint in _AUDIT_CHECKS:
        src = src_pipeline if label in _PIPELINE_ONLY else src_main
        if re.search(pattern, src, re.MULTILINE):
            passed.append(label)
        else:
            failed.append((label, hint))

    return passed, failed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Master Reset — sync winner_odds.json to today's schedule.")
    parser.add_argument("--schedule", metavar="JSON", default=None,
                        help="Force a local schedule JSON instead of live API fetch.")
    parser.add_argument("--output", metavar="PATH", default=str(_DEFAULT_OUTPUT),
                        help=f"Output path for winner_odds.json (default: {_DEFAULT_OUTPUT})")
    args = parser.parse_args()

    output = Path(args.output)
    force_local = Path(args.schedule) if args.schedule else None

    print(f"\n{'=' * 60}")
    print(f"  MUNDIAL MASTER RESET")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 60}")

    exit_code = 0

    # ── PHASE 1: Fetch schedule ───────────────────────────────────────────────
    _header("PHASE 1 — Fetching today's schedule")
    try:
        raw_games, source = fetch_schedule(force_local)
        print(f"  Source : {source}")
        print(f"  Total  : {len(raw_games)} match(es) in schedule")
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return 1

    # ── PHASE 2: Filter today's matches (IDT window) ─────────────────────────
    _header("PHASE 2 — IDT-day matches (prev day 21:00 UTC -> today+30h UTC)")
    today = todays_matches(raw_games)
    now_utc = datetime.now(timezone.utc)
    today_start_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    window_start_utc = today_start_utc - timedelta(hours=3)
    print(
        f"  Clock  : {now_utc.strftime('%Y-%m-%d %H:%M UTC')}  "
        f"(IDT window: {window_start_utc.strftime('%Y-%m-%d %H:%M')} UTC -> "
        f"{(today_start_utc + timedelta(hours=30)).strftime('%Y-%m-%d %H:%M')} UTC)"
    )

    upcoming = [g for g in today if g.get("status") != "final"]
    played   = [g for g in today if g.get("status") == "final"]

    if not today:
        print("  WARNING: No matches found for today.")
        print("           This is expected if there are no WC games today.")
        print("           winner_odds.json will be reset to empty.")
    else:
        for g in today:
            h_key = g.get("home", ""); a_key = g.get("away", "")
            h = _team_name(g, h_key); a = _team_name(g, a_key)
            start = _parse_start(g.get("start_time", ""))
            status = g.get("status", "?")
            suffix = " [ALREADY PLAYED — logged in _played_today, no odds needed]" \
                if status == "final" else ""
            print(f"  [{status:<10}]  {h} vs {a}  @ {start.strftime('%H:%M UTC')}{suffix}")

    # ── PHASE 3: Update winner_odds.json (preserve filled odds) ──────────────
    _header(f"PHASE 3 — Updating {output} (preserving filled odds)")

    new_keys, preserved_keys = reset_winner_odds(today, output)
    blank_keys = [k for k in new_keys if k not in preserved_keys]

    if played:
        print(f"\n  Already played ({len(played)} game(s) — recorded in _played_today):")
        for g in played:
            h = _team_name(g, g.get("home", "")); a = _team_name(g, g.get("away", ""))
            print(f"    ~ {h} vs {a}  [final — no odds entry created]")

    if preserved_keys:
        print(f"\n  Preserved {len(preserved_keys)} entry/entries (odds already filled):")
        for k in preserved_keys:
            print(f"    = {k}  (kept — odds > 0.0)")
    if blank_keys:
        print(f"\n  Added {len(blank_keys)} placeholder entry/entries (fill in before pipeline):")
        for k in blank_keys:
            print(f"    + {k}  (odds: 0.0)")
    if not new_keys:
        print("  (no upcoming matches today — file written with only _note metadata)")

    # ── PHASE 4: Audit main.py ────────────────────────────────────────────────
    _header(f"PHASE 4 — Auditing {_MAIN_PY.name}")
    passed, failed = audit_main(_MAIN_PY)

    for label in passed:
        print(f"  PASS  {label}")

    if failed:
        print()
        for label, hint in failed:
            print(f"  FAIL  {label}")
            print(f"        Fix: {hint}")
        exit_code = 1
    else:
        print(f"\n  All {len(passed)} integration checks passed.")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    audit_status = "ALL PASS" if not failed else f"{len(failed)} FAIL(S)"
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Schedule source  : {source}")
    print(f"  Today's matches  : {len(today)}")
    _blank_count = len(new_keys) - len(preserved_keys)
    print(f"  winner_odds.json : {len(preserved_keys)} preserved, {_blank_count} new (odds=0.0)")
    print(f"  main.py audit    : {audit_status}  ({len(passed)}/{len(passed)+len(failed)} checks)")

    if blank_keys:
        print(f"\n  ACTION REQUIRED:")
        print(f"  Open {output} and fill in")
        print(f"  decimal odds for each match before the pipeline runs.")
        print(f"  Leave at 0.0 to skip O/U EV detection (pipeline still works).")
    elif not new_keys:
        print(f"\n  No WC matches today — winner_odds.json cleared. Pipeline will")
        print(f"  exit cleanly with 'No odds found for today'.")
    else:
        print(f"\n  All odds already filled — no action required.")

    print(f"{'=' * 60}\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
