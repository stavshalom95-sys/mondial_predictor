"""
main.py — Orchestration entry point.

Flow:
  1. Load raw schedule JSON (from scripts/fetch_schedule.py or any compatible source).
  2. parse_world_cup_schedule -> compute matches_remaining -> update MY_CURRENT_STATE.
  3. For each match in get_todays_manual_odds():
       - Skip if already finished (status == "final").
       - odds_converter -> poisson_engine -> strategy_advisor.
  4. format_daily_message -> send_whatsapp_message.
  5. Print and return the message text (useful even when send_notification=False).

Daily manual update required:
  Edit get_todays_manual_odds() with today's decimal odds (~30 seconds work).

Usage:
  python main.py tests/sample_games.json
  python main.py tests/sample_games.json --no-notify
"""
from __future__ import annotations

import argparse
import json
import sys

# Ensure stdout can handle Unicode/emoji on all platforms (e.g. Windows cp1255 terminals)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.scoring_rules import TournamentStage
from core.odds_converter import MatchOdds1X2, OverUnderOdds, remove_overround, remove_overround_ou
from core.poisson_engine import calibrate
from core.strategy_advisor import TournamentContext, recommend
from data.data_pipeline import (
    parse_world_cup_schedule,
    matches_remaining_in_tournament,
    get_match_by_teams,
)
from data.tournament_state import MY_CURRENT_STATE, LONG_TERM_BETS
from notifications.notifier import DailyPick, format_daily_message, send_whatsapp_message


# ---------------------------------------------------------------------------
# *** EDIT THIS DAILY ***  (~30 seconds of work)
# ---------------------------------------------------------------------------

def get_todays_manual_odds() -> dict[tuple[str, str], dict]:
    """
    Map (home_team_name, away_team_name) -> odds + stage info.

    Fill in today's decimal odds from any bookmaker before running.
    Remove or comment out matches not played today.

    Example entry:
      ("Spain", "Saudi Arabia"): {
          "odds_1x2": MatchOdds1X2(home=1.25, draw=6.00, away=13.00),
          "ou_odds":  OverUnderOdds(line=2.5, over=1.80, under=2.00),  # optional
          "stage":    TournamentStage.GROUP_STAGE,
      },
    """
    return {
        ("Spain", "Saudi Arabia"): {
            "odds_1x2": MatchOdds1X2(home=1.25, draw=6.00, away=13.00),
            "ou_odds":  OverUnderOdds(line=2.5, over=1.80, under=2.00),
            "stage":    TournamentStage.GROUP_STAGE,
        },
        # Add more matches here as needed:
        # ("Brazil", "Mexico"): {
        #     "odds_1x2": MatchOdds1X2(home=1.55, draw=3.80, away=5.50),
        #     "stage":    TournamentStage.GROUP_STAGE,
        # },
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_daily_pipeline(
    raw_games_from_api: list[dict],
    send_notification: bool = True,
) -> str:
    """
    Full pipeline: parse schedule, analyse each match, send WhatsApp.
    Returns the formatted message string regardless of send_notification.
    """
    # Step 1: Parse schedule + update matches_remaining
    all_matches = parse_world_cup_schedule(raw_games_from_api)
    remaining   = matches_remaining_in_tournament(all_matches)

    # Update state (in-memory; does not write to file)
    MY_CURRENT_STATE["matches_remaining"] = remaining

    context = TournamentContext(
        my_points         = MY_CURRENT_STATE["my_points"],
        leader_points     = MY_CURRENT_STATE["leader_points"],
        matches_remaining = remaining,
    )

    print(f"[pipeline] Schedule loaded: {len(all_matches)} matches total, {remaining} remaining.")

    # Step 2: Analyse each match with manual odds
    todays_odds = get_todays_manual_odds()
    picks: list[DailyPick] = []

    for (home_team, away_team), cfg in todays_odds.items():
        match = get_match_by_teams(all_matches, home_team, away_team)

        if match is None:
            print(f"[pipeline] WARNING: match '{home_team} vs {away_team}' not found in schedule — skipping.")
            continue

        if match.status == "final":
            print(f"[pipeline] '{home_team} vs {away_team}' already finished — skipping.")
            continue

        stage    = cfg["stage"]
        odds_1x2 = cfg["odds_1x2"]
        ou_odds  = cfg.get("ou_odds")

        # odds -> true probabilities
        true_probs = remove_overround(odds_1x2)
        ou_probs   = remove_overround_ou(ou_odds) if ou_odds else None

        print(
            f"[pipeline] {home_team} vs {away_team} | "
            f"overround={true_probs.overround*100:.1f}% | "
            f"true: H={true_probs.home:.1%} D={true_probs.draw:.1%} A={true_probs.away:.1%}"
        )

        # true probabilities -> Poisson model
        model = calibrate(true_probs, ou_probs)
        print(f"[pipeline]   lam_home={model.lambda_home:.2f}  lam_away={model.lambda_away:.2f}")
        print(f"[pipeline]   top-3: {model.top_n(3)}")

        # Poisson model -> strategy recommendation
        rec = recommend(model, context, stage)
        print(f"[pipeline]   recommendation: {rec.recommended_pick} [{rec.strategy.value}]")

        picks.append(DailyPick(
            home_team      = home_team,
            away_team      = away_team,
            recommendation = rec,
        ))

    if not picks:
        message = "אין משחקים להיום עם יחסי הימורים ידניים."
        print("[pipeline]", message)
        return message

    # Step 3: Format + send
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
    parser = argparse.ArgumentParser(description="Mondial Predictor — daily pipeline")
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
