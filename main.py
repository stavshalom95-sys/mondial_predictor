"""
main.py — Fully automated daily orchestration pipeline.

Zero manual steps required. Run sequence:
  1. Fetch live standings from 365Scores (public endpoint) → override tournament_state.py defaults
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
  GREEN_API_INSTANCE_ID       — green-api.com
  GREEN_API_TOKEN             — green-api.com
  WHATSAPP_RECIPIENT_PHONE    — recipient in international format (e.g. 972501234567)
  (365Scores endpoint is public — no secret needed for standings)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

# Ensure stdout handles Unicode/emoji on all platforms (e.g. Windows cp1255 terminals)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.odds_converter import MatchOdds1X2, remove_overround, remove_overround_ou
from core.poisson_engine import PoissonMatchModel, _build_matrix, calibrate
from core.strategy_advisor import TournamentContext, recommend
from data.data_pipeline import (
    parse_world_cup_schedule,
    matches_remaining_in_tournament,
    get_todays_matches,
)
from data.odds_fetcher import fetch_todays_match_odds, _normalize as normalize_team
from data.scores365_sync import fetch_standings
from data.tournament_state import MY_CURRENT_STATE
from notifications.notifier import DailyPick, format_daily_message, format_lineup_alert, send_whatsapp_message

from config.scoring_rules import TournamentStage
from core.ai_ensemble import enhance
from data.context_fetcher import fetch_match_context
from data.results_fetcher import fetch_yesterday_results
from data.performance_tracker import ingest_results, load_history, save_history, yesterday_stats

_MORNING_PICKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "morning_picks.json")


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

    # ── Step 0: Ingest yesterday's results + build performance report ────────
    rapidapi_key = os.environ.get("RAPIDAPI_KEY", "")
    yesterday_results = fetch_yesterday_results(api_key=rapidapi_key)
    history = load_history()

    if yesterday_results:
        # Load this morning's picks to know what was predicted yesterday
        perf_report: dict | None = None
        if os.path.exists(_MORNING_PICKS_PATH):
            try:
                with open(_MORNING_PICKS_PATH, encoding="utf-8") as f:
                    yesterday_picks = json.load(f)
                history = ingest_results(yesterday_picks, yesterday_results, history)
                save_history(history)
                perf_report = yesterday_stats(history)
            except Exception as exc:
                print(f"[pipeline] Warning: result ingestion failed: {exc}")
                perf_report = None
        else:
            print("[pipeline] morning_picks.json not found — cannot score yesterday's predictions.")
            perf_report = None
    else:
        perf_report = yesterday_stats(history)

    # ── Step 1: Live standings sync ──────────────────────────────────────────
    live_standings = fetch_standings()

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

    picks:        list[DailyPick] = []
    morning_data: list[dict]      = []

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

        # Defense-in-depth: if The Odds API listed teams in the opposite order to
        # football-data.org, the odds_key home doesn't match the schedule home.
        # Swap home/away odds so the Poisson model's "home" always = schedule home.
        if odds_key[0] != normalize_team(match.home_team):
            odds_1x2 = MatchOdds1X2(home=odds_1x2.away, draw=odds_1x2.draw, away=odds_1x2.home)
            print(f"[pipeline]   NOTE: home/away odds swapped — API order differed from schedule")

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

        # ── AI Ensemble: context (injuries, form) + Claude calibration ──────────
        rapidapi_key    = os.environ.get("RAPIDAPI_KEY", "")
        match_ctx       = fetch_match_context(match.home_team, match.away_team, api_key=rapidapi_key)
        context_section = match_ctx.to_prompt_section() if match_ctx else ""

        ai_pick_prob = None
        ai_reasoning = None
        ensemble_pick = enhance(
            home_team       = match.home_team,
            away_team       = match.away_team,
            stage           = stage,
            model           = model,
            context_section = context_section,
        )
        if ensemble_pick:
            ai_pick_prob = ensemble_pick.to_score_prob(model)
            ai_reasoning = ensemble_pick.reasoning

        picks.append(DailyPick(
            home_team      = match.home_team,
            away_team      = match.away_team,
            recommendation = rec,
            ai_pick        = ai_pick_prob,
            ai_reasoning   = ai_reasoning,
        ))
        morning_data.append({
            "date":             date.today().isoformat(),
            "home_team":        match.home_team,
            "away_team":        match.away_team,
            "stage":            stage.value,
            "lambda_home":      model.lambda_home,
            "lambda_away":      model.lambda_away,
            "final_home_goals": (ai_pick_prob.home_goals  if ai_pick_prob else rec.recommended_pick.home_goals),
            "final_away_goals": (ai_pick_prob.away_goals  if ai_pick_prob else rec.recommended_pick.away_goals),
        })

    if not picks:
        msg = "[pipeline] No matches could be analysed today (odds/schedule mismatch or all finished)."
        print(msg)
        if send_notification:
            send_whatsapp_message(msg)
        return msg

    # ── Step 5: Format + send ────────────────────────────────────────────────
    message = format_daily_message(picks, context, perf_report=perf_report)

    if send_notification:
        send_whatsapp_message(message)
    else:
        print("\n--- Message (notification disabled) ---")
        print(message)
        print("--- End of message ---\n")

    save_morning_picks(morning_data)
    return message


# ---------------------------------------------------------------------------
# Morning picks persistence
# ---------------------------------------------------------------------------

def save_morning_picks(morning_data: list) -> None:
    """Persist today's match records to morning_picks.json for the lineup-check run."""
    if not morning_data:
        return
    os.makedirs(os.path.dirname(_MORNING_PICKS_PATH), exist_ok=True)
    try:
        with open(_MORNING_PICKS_PATH, "w", encoding="utf-8") as f:
            json.dump(morning_data, f, indent=2, ensure_ascii=False)
        print(f"[pipeline] Morning picks saved → {_MORNING_PICKS_PATH}")
    except Exception as exc:
        print(f"[pipeline] Warning: could not save morning picks: {exc}")


# ---------------------------------------------------------------------------
# Lineup check pipeline (pre-match delta alert)
# ---------------------------------------------------------------------------

def run_lineup_check_pipeline(send_notification: bool = True) -> None:
    """
    Pre-match run (~60 min before kick-off).
    Loads morning picks, fetches confirmed starting XIs, re-runs Claude.
    Sends a WhatsApp alert ONLY if the AI prediction changed — no spam.
    """
    if not os.path.exists(_MORNING_PICKS_PATH):
        print("[lineup] morning_picks.json not found — run the morning pipeline first.")
        return

    with open(_MORNING_PICKS_PATH, encoding="utf-8") as f:
        morning_data = json.load(f)

    print(f"[lineup] Loaded {len(morning_data)} morning picks.")
    alerts_sent = 0

    for record in morning_data:
        home_team = record["home_team"]
        away_team = record["away_team"]
        stage     = TournamentStage(record["stage"])
        old_h     = record["final_home_goals"]
        old_a     = record["final_away_goals"]

        print(f"\n[lineup] Checking {home_team} vs {away_team}...")

        # Rebuild Poisson model from saved lambda values
        lh, la = record["lambda_home"], record["lambda_away"]
        model = PoissonMatchModel(
            lambda_home = lh,
            lambda_away = la,
            _matrix     = _build_matrix(lh, la),
        )

        # Fetch confirmed starting XIs via API-Football
        rapidapi_key = os.environ.get("RAPIDAPI_KEY", "")
        match_ctx = fetch_match_context(
            home_team, away_team,
            include_lineups = True,
            api_key         = rapidapi_key,
        )

        if match_ctx is None or not match_ctx.lineups_confirmed:
            print(f"[lineup]   Lineups not yet confirmed — skipping.")
            continue

        # Re-run Claude with full context including confirmed XIs
        ensemble_pick = enhance(
            home_team       = home_team,
            away_team       = away_team,
            stage           = stage,
            model           = model,
            context_section = match_ctx.to_prompt_section(),
        )
        if ensemble_pick is None:
            print(f"[lineup]   Claude unavailable — skipping.")
            continue

        new_h = ensemble_pick.chosen_home_goals
        new_a = ensemble_pick.chosen_away_goals

        if (new_h, new_a) == (old_h, old_a):
            print(f"[lineup]   Prediction unchanged ({old_h}-{old_a}) — no alert sent.")
            continue

        # Prediction changed — build and send the delta alert
        print(f"[lineup]   PREDICTION CHANGED: {old_h}-{old_a} → {new_h}-{new_a}")
        alert = format_lineup_alert(
            home_team = home_team,
            away_team = away_team,
            old_home  = old_h,
            old_away  = old_a,
            new_home  = new_h,
            new_away  = new_a,
            reasoning = ensemble_pick.reasoning,
        )

        if send_notification:
            send_whatsapp_message(alert)
        else:
            print("\n--- Lineup Alert (notification disabled) ---")
            print(alert)
            print("--- End of alert ---\n")
        alerts_sent += 1

    print(f"\n[lineup] Done. {alerts_sent} alert(s) sent.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mondial Predictor — fully automated daily pipeline")
    parser.add_argument(
        "games_json",
        nargs="?",
        default=None,
        help="Path to JSON file produced by scripts/fetch_schedule.py (required for morning run)",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip sending WhatsApp notification (print message instead)",
    )
    parser.add_argument(
        "--lineup-check",
        action="store_true",
        help="Pre-match run: fetch confirmed lineups and send alert if prediction changed",
    )
    args = parser.parse_args()

    if args.lineup_check:
        run_lineup_check_pipeline(send_notification=not args.no_notify)
    elif args.games_json:
        with open(args.games_json, encoding="utf-8") as f:
            raw_games = json.load(f)
        run_daily_pipeline(raw_games, send_notification=not args.no_notify)
    else:
        parser.error("games_json is required for the morning run (or use --lineup-check)")


if __name__ == "__main__":
    main()
