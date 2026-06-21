"""
main.py — Fully automated daily orchestration pipeline.

Zero manual steps required. Run sequence:
  1. Fetch live standings from 365Scores (auth cookie) → override tournament_state.py defaults
  2. Fetch today's match odds from The Odds API automatically
  3. Parse WC schedule (from fetch_schedule.py JSON) → get today's matches
  4. Match odds to schedule → run Poisson + game-theory engine per match
  5. Format WhatsApp message → send via Green-API

Usage (called by GitHub Actions):
  python main.py tests/sample_games.json
  python main.py tests/sample_games.json --no-notify

Required GitHub Secrets (env vars):
  FOOTBALL_DATA_API_KEY       — football-data.org (for fetch_schedule.py)
  THE_ODDS_API_KEY            — the-odds-api.com
  SCORE365_AUTH_COOKIE        — browser session cookie from 365Scores
  GREEN_API_INSTANCE_ID       — green-api.com
  GREEN_API_TOKEN             — green-api.com
  WHATSAPP_RECIPIENT_PHONE    — recipient in international format (e.g. 972501234567)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure stdout handles Unicode/emoji on all platforms (e.g. Windows cp1255 terminals)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.odds_converter import remove_overround, remove_overround_ou
from core.poisson_engine import calibrate
from core.strategy_advisor import TournamentContext, recommend
from data.data_pipeline import (
    parse_world_cup_schedule,
    matches_remaining_in_tournament,
    get_todays_matches,
)
from data.odds_fetcher import fetch_todays_match_odds, _normalize as normalize_team
from data.scores365_sync import fetch_standings
from data.tournament_state import MY_CURRENT_STATE
from notifications.notifier import DailyPick, format_daily_message, send_whatsapp_message


# ---------------------------------------------------------------------------
# Team name matching
# ---------------------------------------------------------------------------

def _teams_match(schedule_name: str, odds_key: str) -> bool:
    """
    True if a schedule team name and an odds-dict key refer to the same team.
    Both are normalized via the same _normalize() used in odds_fetcher.
    """
    return normalize_team(schedule_name) == odds_key


def _find_odds_for_match(
    home_team: str,
    away_team: str,
    odds_map: dict[tuple[str, str], dict],
) -> tuple[str, str] | None:
    """
    Return the (home_key, away_key) in odds_map that matches the schedule team names,
    or None if not found. Tries exact normalized match first, then partial-string fallback.
    """
    h_norm = normalize_team(home_team)
    a_norm = normalize_team(away_team)

    # Exact normalized match
    if (h_norm, a_norm) in odds_map:
        return (h_norm, a_norm)

    # Partial fallback: one name contains the other (handles "United States" vs "USA" edge cases)
    for (oh, oa) in odds_map:
        if (h_norm in oh or oh in h_norm) and (a_norm in oa or oa in a_norm):
            return (oh, oa)

    return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_daily_pipeline(
    raw_games_from_api: list[dict],
    send_notification: bool = True,
) -> str:
    """
    Full automated pipeline. Returns formatted message string regardless of send_notification.
    """

    # ── Step 1: Live standings sync ──────────────────────────────────────────
    auth_cookie = os.environ.get("SCORE365_AUTH_COOKIE", "")
    live_standings = fetch_standings(auth_cookie)

    if live_standings:
        MY_CURRENT_STATE["my_points"]     = live_standings["my_points"]
        MY_CURRENT_STATE["leader_points"] = live_standings["leader_points"]
        MY_CURRENT_STATE["leader_name"]   = live_standings["leader_name"]
    else:
        print("[pipeline] Using hardcoded standings from tournament_state.py (365Scores sync unavailable).")

    # ── Step 2: Parse schedule + compute remaining matches ───────────────────
    all_matches = parse_world_cup_schedule(raw_games_from_api)
    remaining   = matches_remaining_in_tournament(all_matches)
    MY_CURRENT_STATE["matches_remaining"] = remaining

    print(f"[pipeline] Schedule: {len(all_matches)} total matches | {remaining} remaining.")

    context = TournamentContext(
        my_points         = MY_CURRENT_STATE["my_points"],
        leader_points     = MY_CURRENT_STATE["leader_points"],
        matches_remaining = remaining,
    )

    # ── Step 3: Auto-fetch today's odds ──────────────────────────────────────
    odds_api_key = os.environ.get("THE_ODDS_API_KEY", "")
    odds_map = fetch_todays_match_odds(odds_api_key)

    if not odds_map:
        msg = "[pipeline] No odds found for today — nothing to analyse. Check THE_ODDS_API_KEY or no WC matches today."
        print(msg)
        if send_notification:
            send_whatsapp_message(msg)
        return msg

    # ── Step 4: Match odds to today's schedule ────────────────────────────────
    todays_matches = get_todays_matches(all_matches)
    print(f"[pipeline] Today's matches in schedule: {len(todays_matches)}")

    picks: list[DailyPick] = []

    for match in todays_matches:
        odds_key = _find_odds_for_match(match.home_team, match.away_team, odds_map)

        if odds_key is None:
            print(f"[pipeline] No odds matched for '{match.home_team} vs {match.away_team}' — skipping.")
            continue

        cfg = odds_map[odds_key]

        if match.status == "final":
            print(f"[pipeline] '{match.home_team} vs {match.away_team}' already finished — skipping.")
            continue

        odds_1x2 = cfg["odds_1x2"]
        ou_odds  = cfg.get("ou_odds")
        # Stage: prefer the one from the schedule (set by football-data.org), fall back to odds_fetcher inference
        stage    = match.stage if match.stage else cfg["stage"]

        # odds → true probabilities
        true_probs = remove_overround(odds_1x2)
        ou_probs   = remove_overround_ou(ou_odds) if ou_odds else None

        print(
            f"[pipeline] {match.home_team} vs {match.away_team} [{stage.value}] | "
            f"overround={true_probs.overround*100:.1f}% | "
            f"H={true_probs.home:.1%} D={true_probs.draw:.1%} A={true_probs.away:.1%}"
        )

        # true probabilities → Poisson model
        model = calibrate(true_probs, ou_probs)
        print(f"[pipeline]   lam_home={model.lambda_home:.2f}  lam_away={model.lambda_away:.2f}")
        print(f"[pipeline]   top-3: {model.top_n(3)}")

        # Poisson model → strategy recommendation
        rec = recommend(model, context, stage)
        print(f"[pipeline]   recommendation: {rec.recommended_pick} [{rec.strategy.value}]")

        picks.append(DailyPick(
            home_team      = match.home_team,
            away_team      = match.away_team,
            recommendation = rec,
        ))

    if not picks:
        msg = "[pipeline] No matches could be analysed today (odds/schedule mismatch or all finished)."
        print(msg)
        if send_notification:
            send_whatsapp_message(msg)
        return msg

    # ── Step 5: Format + send ────────────────────────────────────────────────
    message = format_daily_message(picks, context)

    if send_notification:
        send_whatsapp_message(message)
    else:
        print("\n--- Message (notification disabled) ---")
        print(message)
        print("--- End of message ---\n")

    return message


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mondial Predictor — fully automated daily pipeline")
    parser.add_argument(
        "games_json",
        help="Path to JSON file produced by scripts/fetch_schedule.py",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip sending WhatsApp notification (print message instead)",
    )
    args = parser.parse_args()

    with open(args.games_json, encoding="utf-8") as f:
        raw_games = json.load(f)

    run_daily_pipeline(raw_games, send_notification=not args.no_notify)


if __name__ == "__main__":
    main()
