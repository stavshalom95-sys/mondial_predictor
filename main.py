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
from datetime import date, datetime

# Ensure stdout handles Unicode/emoji on all platforms (e.g. Windows cp1255 terminals)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.odds_converter import MatchOdds1X2, remove_overround, remove_overround_ou
from core.poisson_engine import PoissonMatchModel, _build_matrix, build_dc_matrix, calibrate, calibrate_dc
from core.strategy_advisor import TournamentContext, recommend
from data.data_pipeline import (
    parse_world_cup_schedule,
    matches_remaining_in_tournament,
    get_todays_matches,
)
from data.odds_fetcher import fetch_todays_match_odds, _normalize as normalize_team
from data.scores365_sync import fetch_standings
from data.tournament_state import MY_CURRENT_STATE
from notifications.notifier import DailyPick, format_daily_message, format_lineup_alert, send_whatsapp_message, TOTAL_BANKROLL, DAILY_BUDGET_CAP, LOW_PROB_BUDGET_CAP

from config.scoring_rules import SCORING, TournamentStage
from core.ai_ensemble import enhance, get_ai_offline_reason
from data.context_fetcher import fetch_match_context
from data.backup_scraper import fetch_match_context_espn
from data.results_fetcher import fetch_yesterday_results
from data.performance_tracker import ingest_results, load_history, save_history, yesterday_stats, compute_stats
from data.opta_priors import build_opta_context, get_whatsapp_sentiment_note, opta_tiebreak, get_team_opta
from core.correct_score_predictor import predict as predict_correct_score, get_external_xg, load_external_xg
from notifications.notifier import format_dual_track_section
from core.bias_corrector import build_bias_corrector, build_goal_rate_scaler
from data.fdr_fetcher import fetch_fixture_mu, apply_fdr_modifier
from core.kelly import analyse_match as analyse_match_bets, BetAnalysis, build_ticket, build_probability_ticket, build_confidence_value_ticket, ConfidenceTicket
from core.simulator import simulate
from core.strength_model import build_strength_model, save_wc_priors, load_wc_priors, MIN_BLEND, BLEND_WEIGHT, dynamic_blend_weight, _norm as _sm_norm
from core.calibration import build_calibrator
from core.market_calculator import calculate_all_markets
from data.winner_odds_loader import enrich_picks, get_all_odds, find_match_odds
from data.motivation import load_group_tables, build_match_motivation
from data.stats_collector import build_form_cache, FORM_BLEND_WEIGHT

_DATA_DIR            = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_MORNING_PICKS_PATH  = os.path.join(_DATA_DIR, "morning_picks.json")
_LAST_RUN_PATH       = os.path.join(_DATA_DIR, "last_run.json")
_WC_PRIORS_PATH      = os.path.join(_DATA_DIR, "wc_priors.json")
_EV_LOG_PATH         = os.path.join(_DATA_DIR, "ev_log.json")
_WINNER_ODDS_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "winner_odds.json")
_GROUP_TABLES_PATH   = os.path.join(_DATA_DIR, "group_tables.json")


# ---------------------------------------------------------------------------
# Schedule-derived results (primary fallback for history ingestion)
# ---------------------------------------------------------------------------

def _results_from_schedule(raw_games: list[dict]) -> list[dict]:
    """
    Extract completed match scores directly from the schedule JSON.

    football-data.org marks finished matches with status="final" and includes
    fulltime scores. This gives us results without needing RAPIDAPI_KEY —
    the schedule is already fetched by fetch_schedule.py before main.py runs.

    Returns list of {"home_team": str, "away_team": str, "home_goals": int, "away_goals": int}
    """
    results = []
    for g in raw_games:
        if g.get("status") != "final":
            continue
        teams      = g.get("teams", {})
        home_key   = g.get("home", "")
        away_key   = g.get("away", "")
        score      = g.get("score", {})
        home_goals = score.get(home_key)
        away_goals = score.get(away_key)
        if home_goals is None or away_goals is None:
            continue
        home_name = teams.get(home_key, {}).get("name", home_key)
        away_name = teams.get(away_key, {}).get("name", away_key)
        results.append({
            "home_team":  home_name,
            "away_team":  away_name,
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
        })
    return results


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
# Single-run guard  (prevents double API charges on re-triggers)
# ---------------------------------------------------------------------------

def _already_ran_today() -> bool:
    """
    Return True if the pipeline already completed successfully today (UTC date).
    Reads data/last_run.json written at the end of a successful pipeline run.
    """
    try:
        with open(_LAST_RUN_PATH, encoding="utf-8") as f:
            rec = json.load(f)
        return rec.get("date") == date.today().isoformat()
    except Exception:
        return False


def _save_last_run(picks_generated: int) -> None:
    """Stamp data/last_run.json so re-triggers today are skipped."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    try:
        rec = {
            "date":             date.today().isoformat(),
            "completed_at":     datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "picks_generated":  picks_generated,
        }
        with open(_LAST_RUN_PATH, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        print(f"[pipeline] last_run.json written — guard active for the rest of today.")
    except Exception as exc:
        print(f"[pipeline] Warning: could not write last_run.json: {exc}")


def _append_ev_log(home_team: str, away_team: str, analyses: list[BetAnalysis]) -> None:
    """
    Append Kelly/EV analysis records to data/ev_log.json.
    Idempotent: replaces any existing records for the same (date, home, away, outcome).
    """
    today = date.today().isoformat()
    os.makedirs(_DATA_DIR, exist_ok=True)
    try:
        existing: list[dict] = []
        if os.path.exists(_EV_LOG_PATH):
            with open(_EV_LOG_PATH, encoding="utf-8") as f:
                existing = json.load(f)
    except Exception:
        existing = []

    # Remove stale entries for the same match today (re-run idempotency)
    key = (today, home_team, away_team)
    existing = [
        r for r in existing
        if not (r.get("date") == key[0]
                and r.get("home_team") == key[1]
                and r.get("away_team") == key[2])
    ]

    for a in analyses:
        existing.append({
            "date":           today,
            "home_team":      home_team,
            "away_team":      away_team,
            "outcome":        a.outcome,
            "our_prob":       round(a.our_prob, 4),
            "decimal_odds":   round(a.decimal_odds, 2),
            "implied_prob":   round(a.implied_prob, 4),
            "edge_pct":       round(a.edge_pct, 2),
            "ev_per_unit":    round(a.ev_per_unit, 4),
            "kelly_fraction": round(a.kelly_fraction, 4),
            "half_kelly":     round(a.half_kelly, 4),
            "is_value":       a.is_value,
        })

    try:
        with open(_EV_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[pipeline] Warning: could not write ev_log.json: {exc}")


# ---------------------------------------------------------------------------
# Strength-model helpers
# ---------------------------------------------------------------------------

def get_team_strength(team_name: str, strength_model) -> dict:
    """
    Return {'attack': float, 'defence': float, 'games': int} for team_name.
    Falls back to tournament average when the team has no WC data yet.
    """
    if not strength_model:
        avg = 1.3
        return {"attack": avg, "defence": avg, "games": 0}
    avg = strength_model.avg_goals
    ts  = strength_model._stats.get(_sm_norm(team_name))
    if ts and ts.games:
        return {"attack": ts.attack, "defence": ts.defence, "games": ts.games}
    return {"attack": avg, "defence": avg, "games": 0}


def calculate_lambda(home_team: str, away_team: str, strength_model) -> tuple[float, float]:
    """
    Return (λ_home, λ_away) from Dixon-Coles strength ratings.
    Returns (None, None) when strength model is unavailable.
    """
    if not strength_model:
        return None, None
    return strength_model.lambdas(home_team, away_team)


# ---------------------------------------------------------------------------
# Competition pick helpers
# ---------------------------------------------------------------------------

def _competition_score_pick(
    sim,
    gap: int,
    matches_remaining: int,
    stage: "TournamentStage",
) -> tuple[int, int, str]:
    """
    Return the competition-optimal score pick + status string.

    PRIMARY: Poisson modal (score with highest P(exact) from DC-corrected MC sim).

    EV-MAX OVERRIDE searches within Top-3 scores only and fires when ALL hold:
      1. Candidate is one of the Top-3 most probable scores (hard constraint)
      2. Candidate P(exact) >= MIN_CANDIDATE_PROB_FLOOR  (absolute 10% floor —
         never submit a pick with < 1-in-10 chance of landing exactly)
      3. EV gain vs modal > OVERRIDE_MIN_EV_DELTA  (clear, justified improvement)

    Strategy (trailing, July 2026): gap ≥ 0 → need exact score hits to close gap.
    Top-3 + 10% floor keeps picks grounded in statistical reality; ΔEV threshold
    ensures we only deviate from modal when the scoring structure makes it worth it.
    """
    exact_pts     = SCORING[stage]["exact"]
    direction_pts = SCORING[stage]["direction"]

    MIN_CANDIDATE_PROB_FLOOR = 0.07   # absolute P(exact) floor — no pick < 7%
    # Any positive ΔEV fires the override — aggressive exact-score mode (trailing)
    # Only two suppression conditions: p < 7% floor, or candidate is genuinely worse (ΔEV ≤ 0)

    p_home = sim.p_home
    p_draw = sim.p_draw
    p_away = sim.p_away

    def _ev(h: int, a: int, p_exact: float) -> float:
        p_dir = p_home if h > a else (p_draw if h == a else p_away)
        return p_exact * exact_pts + (p_dir - p_exact) * direction_pts

    modal_h, modal_a = sim.score_grid.most_likely_score()
    modal_p  = sim.score_grid.probs[modal_h][modal_a]
    ev_modal = _ev(modal_h, modal_a, modal_p)

    # Log Top-3 for transparency (search goes to top-5)
    top3 = sim.score_grid.top_scores(3)
    print(
        "[sim] Top-3: "
        + "  ".join(f"{h}-{a}({p:.1%})" for h, a, p in top3)
        + f"  | MC: H={p_home:.1%} D={p_draw:.1%} A={p_away:.1%}"
    )

    # Find best EV candidate within Top-5
    best_h, best_a, best_ev = modal_h, modal_a, ev_modal
    for h, a, p in sim.score_grid.top_scores(5):
        e = _ev(h, a, p)
        if e > best_ev:
            best_ev, best_h, best_a = e, h, a

    _pick_status = "modal"
    if (best_h, best_a) != (modal_h, modal_a):
        candidate_p = sim.score_grid.probs[best_h][best_a]
        ev_delta    = best_ev - ev_modal
        if candidate_p >= MIN_CANDIDATE_PROB_FLOOR and ev_delta > 0:
            _pick_status = f"override ΔEV={ev_delta:+.3f} p={candidate_p:.1%}"
            print(
                f"[ev-pick] Override: {modal_h}-{modal_a}({modal_p:.1%}) "
                f"-> {best_h}-{best_a}({candidate_p:.1%})  ΔEV={ev_delta:+.3f}"
            )
            return best_h, best_a, _pick_status
        else:
            _reasons: list[str] = []
            if candidate_p < MIN_CANDIDATE_PROB_FLOOR:
                _reasons.append(f"p={candidate_p:.1%}<{MIN_CANDIDATE_PROB_FLOOR:.0%}floor")
            if ev_delta <= 0:
                _reasons.append(f"ΔEV={ev_delta:+.3f}≤0")
            _pick_status = f"suppressed ({', '.join(_reasons)})"
            print(
                f"[ev-pick] Suppressed: {best_h}-{best_a}({candidate_p:.1%})  "
                + "  ".join(_reasons)
            )

    return modal_h, modal_a, _pick_status


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_daily_pipeline(
    raw_games_from_api: list[dict],
    send_notification: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> str:
    """
    Full automated pipeline. Returns formatted message string regardless of send_notification.

    force=True   — bypass the single-run guard.
    dry_run=True — skip ALL external API calls (odds, standings, AI, notification).
                   Step 0 (history ingestion + clean_team_name matching) still runs so
                   you can verify it without wasting API quota. Does not write
                   morning_picks.json or last_run.json so repeated dry runs are idempotent.
    """
    # ── Single-run guard ──────────────────────────────────────────────────────
    if not force and _already_ran_today():
        msg = (
            f"[pipeline] Guard: pipeline already completed today "
            f"({date.today().isoformat()}). "
            f"Re-run with --force to override."
        )
        print(msg)
        return msg

    # ── Step 0: Ingest completed results into history ────────────────────────
    #
    # Primary source: schedule JSON from football-data.org (always available).
    # Every finished match already carries status="final" + fulltime score.
    #
    # Supplementary: RapidAPI (fetch_yesterday_results) — adds any matches
    # not yet reflected in the schedule snapshot.

    # Primary: extract all "final" entries from the schedule we just received
    schedule_results = _results_from_schedule(raw_games_from_api)
    print(f"[pipeline] Schedule-based results: {len(schedule_results)} finished match(es).")

    # Supplementary: RapidAPI for yesterday — merge in without duplicating
    rapidapi_key    = os.environ.get("RAPIDAPI_KEY", "")
    api_results     = fetch_yesterday_results(api_key=rapidapi_key)
    seen_pairs      = {(r["home_team"].lower(), r["away_team"].lower()) for r in schedule_results}
    for r in api_results:
        if (r["home_team"].lower(), r["away_team"].lower()) not in seen_pairs:
            schedule_results.append(r)

    combined_results = schedule_results

    # Load accumulated WC priors from previous runs (empty dict on first run)
    _wc_priors = load_wc_priors(_WC_PRIORS_PATH)
    if _wc_priors:
        print(f"[pipeline] Loaded {len(_wc_priors)} team priors from wc_priors.json")

    # R3 — Prior decay: teams with inflated historical priors not confirmed by Opta's
    # 25k simulations are shrunk toward the mean. Applies to teams where wc_prior > 2.0
    # but Opta tournament win% < 6% (i.e. "big name, underwhelming 2026 form").
    # Protected: France, Argentina, Spain, England, Brazil (Opta win >= 6%).
    _PRIOR_DECAY_FLOOR = 2.0    # priors above this are "legacy elite" candidates
    _PRIOR_DECAY_GATE  = 6.0    # Opta win% below this → prior is stale
    _PRIOR_DECAY       = 0.88   # reduce by 12%
    if _wc_priors:
        for _pt, _pv in list(_wc_priors.items()):
            if _pv > _PRIOR_DECAY_FLOOR:
                _opta_entry = get_team_opta(_pt)
                _opta_win   = _opta_entry.get("win", 0) if _opta_entry else 0
                if _opta_win < _PRIOR_DECAY_GATE:
                    _wc_priors[_pt] = round(_pv * _PRIOR_DECAY, 4)
                    print(
                        f"[pipeline] R3 prior decay: {_pt} {_pv:.4f} → {_wc_priors[_pt]:.4f}"
                        f" (Opta win={_opta_win}% < {_PRIOR_DECAY_GATE}%)"
                    )

    # Build strength model from completed WC matches (returns None if < MIN_MATCHES)
    strength_model = build_strength_model(combined_results, external_priors=_wc_priors or None)
    if strength_model:
        print(strength_model.summary())

    # Pre-load external xG data (data/external_xg.json — manually populated from images)
    load_external_xg()
    cs_picks: list = []   # CorrectScorePick per match, collected for dual-track WhatsApp section

    history          = load_history()
    perf_report: dict | None = None

    if not combined_results:
        print("[pipeline] Step 0: no finished matches found in schedule or RapidAPI — skipping ingestion.")
        perf_report = yesterday_stats(history)
    elif not os.path.exists(_MORNING_PICKS_PATH):
        print("[pipeline] Step 0: morning_picks.json not found — skipping ingestion (first run).")
        perf_report = yesterday_stats(history)
    else:
        try:
            with open(_MORNING_PICKS_PATH, encoding="utf-8") as f:
                yesterday_picks = json.load(f)

            # Diagnostic: show what we're trying to match
            pick_date   = yesterday_picks[0].get("date", "?") if yesterday_picks else "?"
            pick_teams  = [(p["home_team"], p["away_team"]) for p in yesterday_picks]
            result_teams = [(r["home_team"], r["away_team"]) for r in combined_results]
            print(f"[pipeline] Step 0: picks in morning_picks.json → {len(yesterday_picks)} for {pick_date}")
            print(f"[pipeline]   pick teams  : {pick_teams}")
            print(f"[pipeline]   result teams: {result_teams}")
            if not any(
                pt[0].lower() in rt[0].lower() or rt[0].lower() in pt[0].lower()
                for pt in pick_teams for rt in result_teams
            ):
                print(
                    "[pipeline] Step 0: NO OVERLAP between pick teams and finished results. "
                    "Tonight's matches haven't finished yet — history will be populated on "
                    "tomorrow's 06:00 UTC run once the schedule shows them as 'final'."
                )

            prev_len = len(history)
            history  = ingest_results(yesterday_picks, combined_results, history)
            new_recs = len(history) - prev_len
            print(f"[pipeline] Step 0: ingested {new_recs} new record(s) → history now has {len(history)} total.")

            if new_recs > 0:
                save_history(history)
            elif len(history) > 0:
                # Pre-existing records but nothing new today — still keep the file intact
                save_history(history)
            else:
                print("[pipeline] Step 0: skipping history.json write — 0 records. "
                      "Check tracker logs above to see which result names the API returned.")

            perf_report = yesterday_stats(history)
        except Exception as exc:
            print(f"[pipeline] Warning: result ingestion failed: {exc}")
            perf_report = yesterday_stats(history)

    # ── Build probability calibrator from WC history ─────────────────────────
    # AI Research Skills: temperature-scaling (cross-pollinated from ML calibration)
    _calibrator = build_calibrator(history)
    print(_calibrator.summary())

    # ── Dynamic blend weight: grows with WC match count ───────────────────────
    _n_completed = len(combined_results) if combined_results else 0
    _dyn_blend   = dynamic_blend_weight(_n_completed)
    print(f"[strength] dynamic blend weight: {_dyn_blend:.1%} ({_n_completed} completed matches)")

    # ── Augment perf_report with all-time tournament stats ───────────────────
    if history:
        _all_time = compute_stats(history)
        if _all_time["total"] > 0:
            if perf_report is None:
                perf_report = {}
            perf_report["all_time_correct"]  = _all_time["correct"]
            perf_report["all_time_total"]    = _all_time["total"]
            perf_report["all_time_hit_rate"] = round(_all_time["correct"] / _all_time["total"], 3)
            if _all_time.get("pnl_nis") is not None:
                perf_report["all_time_pnl"] = _all_time["pnl_nis"]
            if _all_time.get("brier_score") is not None:
                perf_report["brier_score"] = _all_time["brier_score"]

    # ── Step 1: Live standings sync ──────────────────────────────────────────
    _standings_source = "fallback"   # assume fallback until live data confirmed
    if dry_run:
        print("[pipeline] Dry run — skipping standings sync (using hardcoded state).")
    else:
        live_standings = fetch_standings()
        if live_standings:
            MY_CURRENT_STATE["my_points"]      = live_standings["my_points"]
            MY_CURRENT_STATE["leader_points"]  = live_standings["leader_points"]
            MY_CURRENT_STATE["leader_name"]    = live_standings["leader_name"]
            MY_CURRENT_STATE["my_rank"]        = live_standings.get("my_rank", 0)
            MY_CURRENT_STATE["second_name"]    = live_standings.get("second_name", "")
            MY_CURRENT_STATE["second_points"]  = live_standings.get("second_points", 0)
            _standings_source = "live"
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
        standings_source  = _standings_source,
        leader_name       = MY_CURRENT_STATE.get("leader_name", ""),
        my_rank           = MY_CURRENT_STATE.get("my_rank", 0),
        second_name       = MY_CURRENT_STATE.get("second_name", ""),
        second_points     = MY_CURRENT_STATE.get("second_points", 0),
    )
    _gap = max(0, context.leader_points - context.my_points)

    # ── Always print today's schedule (visible in dry-run AND live run) ───────
    todays_matches = get_todays_matches(all_matches)
    print(f"[pipeline] ── TODAY'S MATCHES FROM SCHEDULE ({len(todays_matches)} found) ──")
    for _i, _m in enumerate(todays_matches, 1):
        print(f"[pipeline]   {_i}. {_m.home_team} vs {_m.away_team}  "
              f"[{_m.start_time_utc.strftime('%Y-%m-%d %H:%M UTC')}]  status={_m.status}")
    print(f"[pipeline] ── END MATCH LIST ──────────────────────────────────────")

    # ── Step 3: Auto-fetch today's odds ──────────────────────────────────────
    if dry_run:
        msg = (
            "[pipeline] Dry run complete.\n"
            "  • Schedule fetch:   skipped (used cached tests/sample_games.json)\n"
            "  • Standings sync:   skipped\n"
            "  • Odds API:         skipped\n"
            "  • AI / RapidAPI:    skipped\n"
            "  • Notification:     skipped\n"
            "  • morning_picks:    not overwritten\n"
            "  • last_run.json:    not updated\n"
            "Check the [tracker] lines above to verify history ingestion and clean_team_name matching."
        )
        print(msg)
        return msg

    odds_api_key = os.environ.get("THE_ODDS_API_KEY", "")
    odds_map = fetch_todays_match_odds(odds_api_key)

    if not odds_map:
        print("[pipeline] The Odds API returned no matches — continuing with prior-only model for all matches.")
        # Don't bail out: todays_matches are already known from the schedule.
        # Each match will use the prior-only path (Poisson without market calibration).
        # winner_odds.json may still provide bookmaker context for value-bet detection.

    # ── Step 4: Match odds to today's schedule ────────────────────────────────
    # Cross-check: flag any scheduled match that has no odds entry so the user
    # knows to add a team-name alias to _TEAM_NAME_MAP in data/odds_fetcher.py.
    _missing_odds: list[str] = []
    for _m in todays_matches:
        if _m.status != "final" and _find_odds_for_match(_m.home_team, _m.away_team, odds_map) is None:
            _missing_odds.append(f"{_m.home_team} vs {_m.away_team}")
    if _missing_odds:
        print(f"[pipeline] ⚠️  {len(_missing_odds)} scheduled match(es) have NO odds — will use prior-only model:")
        for _mn in _missing_odds:
            print(f"[pipeline]   ✗ {_mn}  → check _TEAM_NAME_MAP in data/odds_fetcher.py")
    else:
        print(f"[pipeline] ✓ Odds coverage: all {len(todays_matches)} scheduled matches have odds.")

    # todays_matches already computed above (before dry-run exit)

    # Pre-load bookmaker odds once (no-op if winner_odds.json absent)
    _winner_odds_cache = get_all_odds(_WINNER_ODDS_PATH)

    # Pre-load group tables for motivation scoring (no-op if absent)
    _group_tables = load_group_tables(_GROUP_TABLES_PATH)

    # Build rolling team-form cache (last-5 WC games per team, zero extra API calls)
    try:
        _form_cache = build_form_cache(raw_games_from_api)
    except Exception as _fc_exc:
        print(f"[form] Warning: form cache failed: {_fc_exc} — form adjustment skipped.")
        _form_cache = None

    # Build per-team bias corrector from WC prediction history
    try:
        _bias = build_bias_corrector(history)
    except Exception as _bc_exc:
        print(f"[bias] Warning: bias corrector failed: {_bc_exc} — no bias correction applied.")
        _bias = None

    # Build tournament-wide goal-rate scaler (actual vs predicted goals/game)
    try:
        _goal_scaler = build_goal_rate_scaler(history)
    except Exception as _gs_exc:
        print(f"[goal_scale] Warning: goal rate scaler failed: {_gs_exc} — no scaling applied.")
        _goal_scaler = None

    picks:           list[DailyPick] = []
    morning_data:    list[dict]      = []
    no_odds_matches: list            = []   # schedule matches with no bookmaker odds yet

    rapidapi_key = os.environ.get("RAPIDAPI_KEY", "")   # defined once; reused per match

    for match in todays_matches:
        odds_key = _find_odds_for_match(match.home_team, match.away_team, odds_map)

        if odds_key is None:
            if match.status == "final":
                print(f"[pipeline] '{match.home_team} vs {match.away_team}' already finished — skipping.")
                continue

            # ── Prior-only path: Odds API has no data → use winner_odds.json if ──
            # available, else fall back to strength model + FIFA priors.
            # FDR (vice-captain.com) + motivation always applied on top.
            # Every scheduled match gets a prediction — no "No odds yet" gaps.
            _pr_stage = match.stage or TournamentStage.GROUP_STAGE

            # Check winner_odds.json first — manual market odds entered before the run
            _pr_wo_entry = find_match_odds(match.home_team, match.away_team, _winner_odds_cache)
            _pr_has_book  = (
                _pr_wo_entry is not None and
                (_pr_wo_entry.get("odds_home") or 0) > 1.0 and
                (_pr_wo_entry.get("odds_away") or 0) > 1.0
            )

            if _pr_has_book:
                # ── Market-calibrated λ from winner_odds.json ──────────────────
                print(
                    f"[prior] 📒 Using manual odds from winner_odds.json for "
                    f"{match.home_team} vs {match.away_team}"
                )
                _pr_1x2_odds = MatchOdds1X2(
                    home=_pr_wo_entry["odds_home"],
                    draw=_pr_wo_entry["odds_draw"],
                    away=_pr_wo_entry["odds_away"],
                )
                _pr_true_probs = remove_overround(_pr_1x2_odds)
                _pr_cal_model  = calibrate_dc(_pr_true_probs)
                _pr_lh_market  = _pr_cal_model.lambda_home
                _pr_la_market  = _pr_cal_model.lambda_away
                print(
                    f"[prior]   market → lam_home={_pr_lh_market:.2f}  lam_away={_pr_la_market:.2f}"
                    f"  overround={_pr_true_probs.overround*100:.1f}%"
                )

                # Blend with strength model (same weight as odds path)
                _pr_str_lh, _pr_str_la = calculate_lambda(match.home_team, match.away_team, strength_model)
                if _pr_str_lh and strength_model and strength_model.n_matches >= MIN_BLEND:
                    _pr_lh = round((1 - _dyn_blend) * _pr_lh_market + _dyn_blend * _pr_str_lh, 3)
                    _pr_la = round((1 - _dyn_blend) * _pr_la_market + _dyn_blend * _pr_str_la, 3)
                    print(
                        f"[prior]   strength blend ({_dyn_blend:.0%}): "
                        f"H={_pr_lh_market}→{_pr_lh}  A={_pr_la_market}→{_pr_la}"
                    )
                else:
                    _pr_lh, _pr_la = _pr_lh_market, _pr_la_market
                _pr_is_prior_only = False
                _pr_predicted_by  = "manual_odds"
            else:
                # ── Strength model + FIFA priors only ──────────────────────────
                print(
                    f"[prior] ⚠️  NO ODDS — generating prior-only prediction for "
                    f"{match.home_team} vs {match.away_team}"
                )
                _pr_lh, _pr_la = (
                    strength_model.lambdas(match.home_team, match.away_team)
                    if strength_model else (1.30, 1.10)
                )
                _pr_is_prior_only = True
                _pr_predicted_by  = "prior_only"

            # FDR modifier — independent of odds source, always applies
            _pr_fdr = fetch_fixture_mu(match.home_team, match.away_team)
            if _pr_fdr:
                _pr_mx  = _build_matrix(_pr_lh, _pr_la)
                _pr_mdl = PoissonMatchModel(lambda_home=_pr_lh, lambda_away=_pr_la, _matrix=_pr_mx)
                _pr_mdl = apply_fdr_modifier(_pr_mdl, mu_home=_pr_fdr[0], mu_away=_pr_fdr[1])
                _pr_lh, _pr_la = _pr_mdl.lambda_home, _pr_mdl.lambda_away

            # Motivation (KO stage → all multipliers are 1.0, no rotation logic)
            _pr_motiv = build_match_motivation(
                match.home_team, match.away_team, _group_tables, combined_results,
                is_knockout=(_pr_stage != TournamentStage.GROUP_STAGE),
            )
            _pr_lh = round(_pr_lh * _pr_motiv.home.lambda_multiplier, 3)
            _pr_la = round(_pr_la * _pr_motiv.away.lambda_multiplier, 3)

            # Apply tournament-wide goal-rate scaling (same scaler as main path)
            if _goal_scaler is not None and abs(_goal_scaler.scale - 1.0) >= 0.005:
                _pr_lh = _goal_scaler.apply(_pr_lh)
                _pr_la = _goal_scaler.apply(_pr_la)

            print(f"[prior]   λ  home={_pr_lh}  away={_pr_la}")

            # Build model + simulate
            _pr_matrix = _build_matrix(_pr_lh, _pr_la)
            _pr_model  = PoissonMatchModel(lambda_home=_pr_lh, lambda_away=_pr_la, _matrix=_pr_matrix)
            _pr_sim    = simulate(_pr_lh, _pr_la)
            _pr_modal_h, _pr_modal_a = _pr_sim.score_grid.most_likely_score()
            _pr_sh, _pr_sa, _pr_pick_status = _competition_score_pick(_pr_sim, _gap, context.matches_remaining, _pr_stage)
            _pr_top3 = [{"h": h, "a": a, "p": round(p, 4)} for h, a, p in _pr_sim.score_grid.top_scores(3)]

            # Sub-markets (for O/U bullet)
            _pr_markets = None
            try:
                _pr_markets = calculate_all_markets(
                    _pr_lh, _pr_la,
                    home_team=match.home_team,
                    away_team=match.away_team,
                )
            except Exception:
                pass
            _pr_bullets: list[str] = [
                "📊 Calibrated from winner_odds.json (Odds API unavailable)"
                if _pr_has_book else
                "⚠️ אין מחירים — תחזית מבוססת מודל בלבד"
            ]
            _pr_ratio = _pr_lh / _pr_la if _pr_la > 0.01 else 1.0
            _pr_lstr  = f"λ H={_pr_lh:.1f} / A={_pr_la:.1f}"
            if _pr_ratio >= 1.6:
                _pr_bullets.append(f"⚡ {match.home_team} dominant ({_pr_ratio:.1f}×, {_pr_lstr})")
            elif _pr_ratio >= 1.2:
                _pr_bullets.append(f"📈 {match.home_team} has the edge ({_pr_lstr})")
            elif _pr_ratio <= 0.625:
                _pr_bullets.append(f"⚡ {match.away_team} dominant ({round(1/_pr_ratio,1)}×, {_pr_lstr})")
            elif _pr_ratio <= 0.833:
                _pr_bullets.append(f"📈 {match.away_team} has the edge ({_pr_lstr})")
            else:
                _pr_bullets.append(f"⚖️ Even match ({_pr_lstr})")
            _PR_STATUS_HE = {
                "must_win":               "חייב לנצח",
                "need_draw":              "צריך תיקו",
                "qualified_secure_1st":   "מובטח ראשון — עשוי לנוח שחקנים",
                "qualified_top_seed_fight": "נאבק על גרעין ראשון — הרכב מלא",
                "qualified":              "מוסמך — עשוי להחליף שחקנים",
                "eliminated":             "מודח — מוריד עצימות",
            }
            if not _pr_motiv.is_trivial():
                for _prt, _prm in (
                    (match.home_team, _pr_motiv.home),
                    (match.away_team, _pr_motiv.away),
                ):
                    if abs(_prm.lambda_multiplier - 1.0) >= 0.05:
                        _prs = _PR_STATUS_HE.get(_prm.qualification_status, _prm.qualification_status)
                        _pr_bullets.append(f"🎯 {_prt}: {_prs} (עצימות ×{_prm.lambda_multiplier:.2f})")
            if _pr_markets and _pr_markets.ou.get(2.5):
                _pr_ou = _pr_markets.ou[2.5]
                _pr_bullets.append(
                    f"⚽ צפי {_pr_ou['expected_goals']:.1f} גולים — "
                    f"Over 2.5: {_pr_ou['p_over']:.0%} | Under 2.5: {_pr_ou['p_under']:.0%}"
                )

            _pr_rec = recommend(_pr_model, context, _pr_stage)
            _pr_show_ctx = (
                _pr_stage != TournamentStage.GROUP_STAGE or
                _pr_motiv.home.played >= 2 or _pr_motiv.away.played >= 2
            )
            picks.append(DailyPick(
                home_team      = match.home_team,
                away_team      = match.away_team,
                recommendation = _pr_rec,
                market_data    = _pr_markets,
                tournament_context_lines = (
                    _pr_motiv.to_whatsapp_lines() or None
                ) if _pr_show_ctx else None,
                why_bullets    = _pr_bullets,
                logic_chain    = (
                    (
                        f"Manual odds (winner_odds.json): λ H={_pr_lh:.2f}/A={_pr_la:.2f}"
                        + (f" + FDR(μ={_pr_fdr[0]:.2f}/{_pr_fdr[1]:.2f})" if _pr_fdr else "")
                    ) if _pr_has_book else (
                        f"Prior (no odds): λ H={_pr_lh:.2f}/A={_pr_la:.2f} "
                        f"— strength model + FIFA priors"
                        + (f" + FDR(μ={_pr_fdr[0]:.2f}/{_pr_fdr[1]:.2f})" if _pr_fdr else "")
                    )
                ),
                sim_score_home = _pr_sh,
                sim_score_away = _pr_sa,
                sim_p_home     = round(_pr_sim.p_home,         4),
                sim_p_draw     = round(_pr_sim.p_draw,         4),
                sim_p_away     = round(_pr_sim.p_away,         4),
                poisson_p_home = round(_pr_sim.poisson_p_home, 4),
                poisson_p_draw = round(_pr_sim.poisson_p_draw, 4),
                poisson_p_away = round(_pr_sim.poisson_p_away, 4),
                lambda_home    = round(_pr_lh, 3),
                lambda_away    = round(_pr_la, 3),
                is_knockout    = (_pr_stage != TournamentStage.GROUP_STAGE),
                prior_only     = _pr_is_prior_only,
                sim_top3       = _pr_top3,
                pick_status    = _pr_pick_status,
            ))
            cs_picks.append(predict_correct_score(
                match.home_team, match.away_team, _pr_sim,
                external_xg=get_external_xg(match.home_team, match.away_team),
                is_knockout=(_pr_stage != TournamentStage.GROUP_STAGE),
            ))
            picks[-1].correct_score_pick = cs_picks[-1]
            morning_data.append({
                "date":          date.today().isoformat(),
                "home_team":     match.home_team,
                "away_team":     match.away_team,
                "stage":         _pr_stage.value,
                "lambda_home":       _pr_lh,   # prior path: market-cal absent; these ARE the final λ
                "lambda_away":       _pr_la,
                "final_lambda_home": _pr_lh,
                "final_lambda_away": _pr_la,
                "final_home_goals": _pr_sh,
                "final_away_goals": _pr_sa,
                "sim_p_home":    round(_pr_sim.p_home, 4),
                "sim_p_draw":    round(_pr_sim.p_draw, 4),
                "sim_p_away":    round(_pr_sim.p_away, 4),
                "sim_top3":      _pr_top3,
                "pick_status":   _pr_pick_status,
                "prior_only":    _pr_is_prior_only,
                "variance_mode": (_pr_sh != _pr_modal_h or _pr_sa != _pr_modal_a),
                "is_knockout":   (_pr_stage != TournamentStage.GROUP_STAGE),
                "predicted_by":  _pr_predicted_by,
            })
            no_odds_matches.append(match)   # kept for reference; not shown as "no odds" in report
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

        # true probabilities → Poisson model (DC-corrected matrix)
        model = calibrate_dc(true_probs, ou_probs)
        print(f"[pipeline]   lam_home={model.lambda_home:.2f}  lam_away={model.lambda_away:.2f}")

        # FDR Strength Modifier: blend calibrated lambdas with vice-captain.com mu values
        fdr_mu = fetch_fixture_mu(match.home_team, match.away_team)
        if fdr_mu is not None:
            model = apply_fdr_modifier(model, mu_home=fdr_mu[0], mu_away=fdr_mu[1])

        print(f"[pipeline]   top-3: {model.top_n(3)}")

        # ── Strength-model λ blending ────────────────────────────────────────
        str_lh, str_la = calculate_lambda(match.home_team, match.away_team, strength_model)
        _lam_h_market = model.lambda_home   # snapshot: pure market-calibrated λ (for logic chain)
        _lam_a_market = model.lambda_away

        if str_lh and strength_model and strength_model.n_matches >= MIN_BLEND:
            lam_h = round((1 - _dyn_blend) * model.lambda_home + _dyn_blend * str_lh, 3)
            lam_a = round((1 - _dyn_blend) * model.lambda_away + _dyn_blend * str_la, 3)
            print(
                f"[strength] blended λ: H={model.lambda_home}→{lam_h}  "
                f"A={model.lambda_away}→{lam_a}  (str={str_lh},{str_la})"
            )
        else:
            lam_h, lam_a = model.lambda_home, model.lambda_away
        _lam_h_post_strength = lam_h   # snapshot after strength blend
        _lam_a_post_strength = lam_a

        # ── Team Form Adjustment (last-5 WC rolling average) ─────────────────
        # Formula: form_scale = team.goals_scored_avg / tournament_avg
        #          lam_final  = (1 - w) × lam  +  w × lam × form_scale
        # where w = FORM_BLEND_WEIGHT (15 %). Falls back gracefully when a
        # team has no WC data yet (n_games == 0 → no adjustment applied).
        _h_form = _a_form = None   # initialized here so AI context block can access them
        if _form_cache is not None:
            _t_avg  = _form_cache.tournament_avg or 1.32
            _h_form = _form_cache.get(match.home_team)
            _a_form = _form_cache.get(match.away_team)
            if _h_form and _h_form.n_games > 0:
                _h_scale = _h_form.goals_scored_avg / _t_avg
                lam_h = round((1 - FORM_BLEND_WEIGHT) * lam_h + FORM_BLEND_WEIGHT * lam_h * _h_scale, 3)
            if _a_form and _a_form.n_games > 0:
                _a_scale = _a_form.goals_scored_avg / _t_avg
                lam_a = round((1 - FORM_BLEND_WEIGHT) * lam_a + FORM_BLEND_WEIGHT * lam_a * _a_scale, 3)
            _h_g = f"{_h_form.goals_scored_avg:.2f}" if (_h_form and _h_form.n_games > 0) else "N/A"
            _a_g = f"{_a_form.goals_scored_avg:.2f}" if (_a_form and _a_form.n_games > 0) else "N/A"
            _h_n = _h_form.n_games if _h_form else 0
            _a_n = _a_form.n_games if _a_form else 0
            print(
                f"[form] {match.home_team}: {_h_g}g/game (last {_h_n})  |  "
                f"{match.away_team}: {_a_g}g/game (last {_a_n})  →  "
                f"lam_h={lam_h}  lam_a={lam_a}"
            )

        _lam_h_post_form = lam_h   # snapshot after form adjustment
        _lam_a_post_form = lam_a

        # ── Per-team Bias Correction (history-driven λ offset) ───────────────
        if _bias is not None:
            _h_off = _bias.get_offset(match.home_team)
            _a_off = _bias.get_offset(match.away_team)
            if _h_off != 0.0:
                lam_h = round(max(0.1, lam_h + _h_off), 3)
                print(f"[bias] {match.home_team} offset {_h_off:+.3f} → lam_h={lam_h}")
            if _a_off != 0.0:
                lam_a = round(max(0.1, lam_a + _a_off), 3)
                print(f"[bias] {match.away_team} offset {_a_off:+.3f} → lam_a={lam_a}")
        _lam_h_post_bias = lam_h   # snapshot after bias correction
        _lam_a_post_bias = lam_a

        # ── Global Goal-Rate Scaling (tournament-wide λ correction) ──────────
        if _goal_scaler is not None and abs(_goal_scaler.scale - 1.0) >= 0.005:
            _old_h, _old_a = lam_h, lam_a
            lam_h = _goal_scaler.apply(lam_h)
            lam_a = _goal_scaler.apply(lam_a)
            print(
                f"[goal_scale] ×{_goal_scaler.scale:.3f}: "
                f"H={_old_h}→{lam_h}  A={_old_a}→{lam_a}"
            )

        # ── Pre-match context (API-Football — injuries, form, goals stats, H2H) ──
        # Fetched here so the stats feed the λ adjustment below; result is
        # also reused unchanged by the AI ensemble prompt later in the loop.
        match_ctx = fetch_match_context(match.home_team, match.away_team, api_key=rapidapi_key)

        # ── Tournament Motivation (group-stage only — KO always runs at full intensity) ──
        _match_motivation = build_match_motivation(
            match.home_team, match.away_team, _group_tables, combined_results,
            is_knockout=(stage != TournamentStage.GROUP_STAGE),
        )
        if not _match_motivation.is_trivial():
            _old_h, _old_a = lam_h, lam_a
            lam_h = round(lam_h * _match_motivation.home.lambda_multiplier, 3)
            lam_a = round(lam_a * _match_motivation.away.lambda_multiplier, 3)
            print(
                f"[motivation] {match.home_team} ({_match_motivation.home.qualification_status}) "
                f"×{_match_motivation.home.lambda_multiplier:.2f}: {_old_h}→{lam_h}  |  "
                f"{match.away_team} ({_match_motivation.away.qualification_status}) "
                f"×{_match_motivation.away.lambda_multiplier:.2f}: {_old_a}→{lam_a}"
            )
        else:
            print(
                f"[motivation] No adjustment for {match.home_team} vs {match.away_team} "
                f"(status: {_match_motivation.home.qualification_status} / "
                f"{_match_motivation.away.qualification_status})"
            )

        # ── API-Football Stats Adjustment (defence concede rate + H2H) ────────
        # Two signals genuinely new vs. the existing form/strength/bias blocks:
        #   1. Opponent's goals_conceded_avg → how "leaky" is the defence we're
        #      attacking? (Dixon-Coles uses WC history only — this uses last-5 data)
        #   2. H2H over25_rate → historical goal-scoring pattern in this fixture.
        # Weight: 15 % per signal — gentle nudge; market Poisson baseline dominates.
        # Each raw ratio is clamped to [0.88, 1.15] before blending.
        _STATS_W   = 0.15
        _T_AVG_API = 1.52   # WC 2026 running average goals/team/game
        _stats_tag = ""     # logged in logic chain if any adjustment is made

        if match_ctx is not None:
            _raw_h = 1.0   # multiplier that will be blended into lam_h
            _raw_a = 1.0   # multiplier that will be blended into lam_a

            # Signal 1 — defence quality: porous away defence → more home goals
            if match_ctx.away_goals_conceded_avg is not None:
                _raw_h *= max(0.88, min(1.15, match_ctx.away_goals_conceded_avg / _T_AVG_API))
            # Signal 1 — defence quality: porous home defence → more away goals
            if match_ctx.home_goals_conceded_avg is not None:
                _raw_a *= max(0.88, min(1.15, match_ctx.home_goals_conceded_avg / _T_AVG_API))

            # Signal 2 — H2H goal pattern: high-scoring history nudges both λ up
            _h2h_mult = 1.0
            if match_ctx.h2h_over25_rate is not None:
                if match_ctx.h2h_over25_rate >= 0.6:
                    _h2h_mult = 1.05   # over 60 % of H2H games had 3+ goals
                elif match_ctx.h2h_over25_rate <= 0.3:
                    _h2h_mult = 0.95   # under 30 % → low-scoring H2H history

            # Blend: new signal at 15 % weight
            _blend_h = (1 - _STATS_W) + _STATS_W * _raw_h * _h2h_mult
            _blend_a = (1 - _STATS_W) + _STATS_W * _raw_a * _h2h_mult

            if abs(_blend_h - 1.0) >= 0.005 or abs(_blend_a - 1.0) >= 0.005:
                _pre_h, _pre_a = lam_h, lam_a
                lam_h = round(max(0.1, lam_h * _blend_h), 3)
                lam_a = round(max(0.1, lam_a * _blend_a), 3)
                _stats_tag = f"Stats×{_blend_h:.3f}/{_blend_a:.3f}"
                print(
                    f"[api-stats] {match.home_team} vs {match.away_team}: "
                    f"def_h={_raw_h:.3f}  def_a={_raw_a:.3f}  h2h={_h2h_mult:.2f} "
                    f"→ lam_h {_pre_h}→{lam_h}  lam_a {_pre_a}→{lam_a}"
                )
            else:
                print(f"[api-stats] No significant adjustment for "
                      f"{match.home_team} vs {match.away_team}")
        else:
            print(f"[api-stats] Context unavailable — stats adjustment skipped")

        # KO intensity boost REMOVED (Measurement-First protocol June 2026).
        # Market odds already price knockout intensity — uncalibrated ×1.20 produced
        # inflated modal scores (e.g. France 4-1 instead of true modal 2-0).
        if stage != TournamentStage.GROUP_STAGE:
            print(
                f"[motivation] Knockout stage ({stage.value}) — no λ boost applied "
                f"(market-calibrated baseline: lam_h={lam_h}  lam_a={lam_a})"
            )

        # ── Logic chain (visible in WhatsApp — shows which factors moved λ) ──
        # Format: each adjustment that moved λH by ≥ 1% or any absolute change is shown.
        # Percentages are relative to the market-calibrated baseline.
        def _pct(before: float, after: float) -> str:
            if before == 0:
                return ""
            delta = (after - before) / before * 100
            return f"{delta:+.0f}%" if abs(delta) >= 0.5 else ""

        _chain_steps: list[str] = [f"Mkt {_lam_h_market:.2f}/{_lam_a_market:.2f}"]
        _str_tag = _pct(_lam_h_market, _lam_h_post_strength)
        if _str_tag:
            _chain_steps.append(f"Str{_str_tag}")
        _form_tag = _pct(_lam_h_post_strength, _lam_h_post_form)
        if _form_tag:
            _chain_steps.append(f"Form{_form_tag}")
        _bias_tag = _pct(_lam_h_post_form, _lam_h_post_bias)
        if _bias_tag:
            _chain_steps.append(f"Bias{_bias_tag}")
        _motiv_mult = _match_motivation.home.lambda_multiplier
        if abs(_motiv_mult - 1.0) >= 0.01:
            _chain_steps.append(f"Motiv×{_motiv_mult:.2f}")
        if _stats_tag:
            _chain_steps.append(_stats_tag)
        _chain_steps.append(f"→ {lam_h:.2f}/{lam_a:.2f}")

        _total_h_pct = (_pct(_lam_h_market, lam_h) or "+0%")
        _logic_chain = " | ".join(_chain_steps) + f"  [{_total_h_pct} total vs market]"
        print(f"[chain] {match.home_team} vs {match.away_team}: {_logic_chain}")

        # ── Monte Carlo Simulation (primary prediction engine) ───────────────
        sim = simulate(lam_h, lam_a)
        _modal_h, _modal_a = sim.score_grid.most_likely_score()
        _sim_h, _sim_a, _sim_pick_status = _competition_score_pick(sim, _gap, context.matches_remaining, stage)
        _sim_top3 = [{"h": h, "a": a, "p": round(p, 4)} for h, a, p in sim.score_grid.top_scores(3)]
        _sim_score_pct  = sim.score_grid.probs[_sim_h][_sim_a]
        print(
            f"[sim] Poisson (analytical): "
            f"H={sim.poisson_p_home:.1%}  D={sim.poisson_p_draw:.1%}  A={sim.poisson_p_away:.1%}"
        )
        print(
            f"[sim] Monte Carlo (n={sim.n_sims:,}): "
            f"H={sim.p_home:.1%}  D={sim.p_draw:.1%}  A={sim.p_away:.1%}"
        )
        _modal_label = f"{_modal_h}-{_modal_a}" if (_sim_h != _modal_h or _sim_a != _modal_a) else "same"
        print(f"[sim] Competition pick: {_sim_h}-{_sim_a}  ({_sim_score_pct:.1%})  [modal: {_modal_label}]")
        # Compare sim vs market (true_probs already computed above)
        _VALUE_THRESHOLD      = 0.05
        _HIGH_VALUE_THRESHOLD = 0.20
        edge_h = sim.p_home - true_probs.home
        edge_d = sim.p_draw - true_probs.draw
        edge_a = sim.p_away - true_probs.away
        sim_value_bet: str | None = None
        if edge_h >= _VALUE_THRESHOLD:
            sim_value_bet = "home"
            tag = "⭐ HIGH-VALUE" if edge_h >= _HIGH_VALUE_THRESHOLD else "\U0001f525 VALUE"
            print(f"[sim] {tag} (Home): sim={sim.p_home:.1%} vs mkt={true_probs.home:.1%} edge={edge_h:+.1%}")
        elif edge_a >= _VALUE_THRESHOLD:
            sim_value_bet = "away"
            tag = "⭐ HIGH-VALUE" if edge_a >= _HIGH_VALUE_THRESHOLD else "\U0001f525 VALUE"
            print(f"[sim] {tag} (Away): sim={sim.p_away:.1%} vs mkt={true_probs.away:.1%} edge={edge_a:+.1%}")
        elif edge_d >= _VALUE_THRESHOLD:
            sim_value_bet = "draw"
            tag = "⭐ HIGH-VALUE" if edge_d >= _HIGH_VALUE_THRESHOLD else "\U0001f525 VALUE"
            print(f"[sim] {tag} (Draw): sim={sim.p_draw:.1%} vs mkt={true_probs.draw:.1%} edge={edge_d:+.1%}")

        # Edge value passed to ensemble to trigger value-priority prompt when edge ≥ 20%
        _edge_map = {"home": edge_h, "draw": edge_d, "away": edge_a}
        _active_edge = _edge_map.get(sim_value_bet or "", 0.0)

        # ── Sub-market probabilities (O/U, AH, BTTS) ───────────────────────
        # Wrapped in try/except: a market calc failure must never abort the
        # pipeline or suppress the WhatsApp notification for this match.
        _match_label = f"{match.home_team} vs {match.away_team}"
        markets = None
        print(f"[markets] DEBUG: entering market calc for {_match_label} (λ={lam_h:.3f}/{lam_a:.3f})")
        try:
            markets = calculate_all_markets(
                lam_h, lam_a,
                home_team=match.home_team,
                away_team=match.away_team,
            )
            print(markets.summary())
        except Exception as _mkt_exc:
            import traceback
            print(
                f"[markets] WARNING: calculation failed for "
                f"{_match_label} — {type(_mkt_exc).__name__}: {_mkt_exc}"
            )
            traceback.print_exc()
        print(f"[markets] DEBUG: markets object is {'SET' if markets is not None else 'NONE (calc failed)'}")

        # ── Sum-goals value bet detection (bookmaker bracket odds vs model) ─
        sg_value_bet: str | None = None
        _match_entry = find_match_odds(match.home_team, match.away_team, _winner_odds_cache)
        if _match_entry and markets:
            _SG_KEY_MAP = {"0-1": "sg_01", "2-3": "sg_23", "+4": "sg_4plus"}
            for _bracket, _model_p in markets.sum_goals.items():
                _book_odds = _match_entry.get(_SG_KEY_MAP[_bracket], 0.0) or 0.0
                if _book_odds > 1.0:
                    _ev = _model_p * _book_odds - 1
                    if _ev >= 0.05:
                        sg_value_bet = _bracket
                        print(
                            f"[sg_ev] \U0001f525 VALUE {_bracket} goals: "
                            f"model={_model_p:.1%}  book={_book_odds}  EV={_ev:+.1%}"
                        )
                        break

        # ── Apply temperature-scaling calibration to Poisson probabilities ──
        # Corrects systematic over/underconfidence using historical outcomes.
        # Only active when ≥20 history records exist (calibrator.summary() shows T).
        _ph_cal, _pd_cal, _pa_cal = _calibrator.calibrate(
            model.p_home_win(), model.p_draw(), model.p_away_win()
        )
        if abs(_ph_cal - model.p_home_win()) > 0.005:
            print(
                f"[calibration] {match.home_team} vs {match.away_team}: "
                f"H {model.p_home_win():.1%}→{_ph_cal:.1%}  "
                f"D {model.p_draw():.1%}→{_pd_cal:.1%}  "
                f"A {model.p_away_win():.1%}→{_pa_cal:.1%}  (T={_calibrator.temperature:.3f})"
            )

        # ── Kelly / Value Bet analysis ──────────────────────────────────────
        # Rebuild model from final motivation/strength/form-adjusted lambdas.
        # Using the original `model` (market-calibrated, line 708) gives draw
        # probabilities before the motivation multipliers are applied (e.g. an
        # eliminated team gets ×0.9, dropping its draw prob from ~12% to ~7%).
        # That pre-adjustment probability triggers false "Draw value" signals.
        _final_kelly_model = PoissonMatchModel(
            lambda_home = lam_h,
            lambda_away = lam_a,
            _matrix     = build_dc_matrix(lam_h, lam_a),
        )
        kelly_analyses = analyse_match_bets(_final_kelly_model, odds_1x2)
        _append_ev_log(match.home_team, match.away_team, kelly_analyses)
        value_bets = [a for a in kelly_analyses if a.is_value]
        if value_bets:
            for vb in value_bets:
                print(
                    f"[kelly] VALUE BET: {vb.outcome} | odds={vb.decimal_odds:.2f} "
                    f"| edge={vb.edge_pct:+.1f}% | EV={vb.ev_per_unit:+.1%} "
                    f"| half-Kelly={vb.half_kelly:.1%}"
                )

        # Compute recommended NIS stake (mirrors notifier.py logic) for P&L tracking
        _kvb = value_bets[0] if value_bets else None
        if _kvb:
            _kvb_stake_raw = _kvb.half_kelly * TOTAL_BANKROLL
            _kvb_stake = (
                min(_kvb_stake_raw, DAILY_BUDGET_CAP)
                if _kvb.our_prob >= 0.40
                else min(_kvb_stake_raw / 4, LOW_PROB_BUDGET_CAP)
            )
        else:
            _kvb_stake = None

        # Poisson model → strategy recommendation
        rec = recommend(model, context, stage)
        print(f"[pipeline]   recommendation: {rec.recommended_pick} [{rec.strategy.value}]")

        # ── AI Ensemble — three-tier context chain ───────────────────────────────
        #
        #   P1 — RapidAPI (RAPIDAPI_KEY)    injuries + form + lineups
        #   P2 — ESPN public feed (no key)  WC form string + record + top scorer
        #   P3 — Internal (schedule JSON)   goal averages + bias notes   ← always
        #
        #  P1 is tried first; if empty, P2 is tried; P3 is always appended.
        #  Source is labelled in [brackets] and prepended to ai_reasoning.

        _context_sources: list[str] = []

        # P1: RapidAPI — injuries, form, goals stats, H2H
        # match_ctx already fetched before the λ-adjustment block above; reuse here.
        context_section = match_ctx.to_prompt_section() if match_ctx else ""
        if context_section.strip():
            _context_sources.append("RapidAPI")

        # P2: ESPN public feed — WC form string + W-D-L record + top scorer
        _espn_ctx: object = None
        if not context_section.strip():
            _espn_ctx = fetch_match_context_espn(match.home_team, match.away_team)
            if _espn_ctx is not None:
                _espn_text = _espn_ctx.to_prompt_section()
                if _espn_text.strip():
                    context_section = _espn_text
                    _context_sources.append("ESPN-public")

        # P3: Internal stats (schedule-derived, always built) ─────────────────
        _t_avg_ctx  = (_form_cache.tournament_avg if _form_cache else None) or 1.52
        _form_lines = []
        for _team, _form in ((match.home_team, _h_form), (match.away_team, _a_form)):
            if _form and _form.n_games > 0:
                _vs_avg = _form.goals_scored_avg / _t_avg_ctx
                _trend  = "above" if _vs_avg > 1.05 else ("below" if _vs_avg < 0.95 else "at")
                _form_lines.append(
                    f"{_team} WC form (last {_form.n_games} game{'s' if _form.n_games > 1 else ''}): "
                    f"scored {_form.goals_scored_avg:.2f}/g, conceded {_form.goals_conceded_avg:.2f}/g "
                    f"({_trend} tournament avg of {_t_avg_ctx:.2f})"
                )
            else:
                _form_lines.append(f"{_team}: no WC matches played yet (tournament debut)")
        if _bias is not None:
            for _team in (match.home_team, match.away_team):
                _off = _bias.get_offset(_team)
                if _off != 0.0:
                    _dir = "under-predicted" if _off > 0 else "over-predicted"
                    _form_lines.append(
                        f"[Model note] {_team} goals systematically {_dir} "
                        f"in past matches; λ adjusted {_off:+.2f} to compensate."
                    )
        _form_context_section = "\n".join(_form_lines)
        if _form_context_section.strip():
            _context_sources.append("Internal")

        # Merge all context — P1/P2 first, P3 always appended
        _merged_context = "\n\n".join(
            s for s in [context_section, _form_context_section] if s.strip()
        )

        _src_label = "/".join(_context_sources) if _context_sources else "Internal"
        print(f"[context] Data sources active for {match.home_team} vs {match.away_team}: {_src_label}")

        # ── Why Bullets — 3-5 short reasons shown in WhatsApp ───────────────
        _why_bullets: list[str] = []

        # 1. λ attack ratio (how lopsided the match is)
        _lam_ratio = round(lam_h / lam_a, 2) if lam_a > 0.01 else 1.0
        _lam_str   = f"λ H={lam_h:.1f} / A={lam_a:.1f}"
        if _lam_ratio >= 1.6:
            _why_bullets.append(f"⚡ {match.home_team} dominant ({_lam_ratio:.1f}× stronger, {_lam_str})")
        elif _lam_ratio >= 1.2:
            _why_bullets.append(f"📈 {match.home_team} has the edge ({_lam_str})")
        elif _lam_ratio <= 0.625:
            _why_bullets.append(f"⚡ {match.away_team} dominant ({round(1/_lam_ratio, 1)}× stronger, {_lam_str})")
        elif _lam_ratio <= 0.833:
            _why_bullets.append(f"📈 {match.away_team} has the edge ({_lam_str})")
        else:
            _why_bullets.append(f"⚖️ Even match ({_lam_str})")

        # 2. WC form (only when a team scored noticeably above/below tournament avg)
        _t_avg_why = (_form_cache.tournament_avg if _form_cache else None) or 1.52
        for _wt, _wf in ((match.home_team, _h_form), (match.away_team, _a_form)):
            if _wf and _wf.n_games > 0:
                if _wf.goals_scored_avg > _t_avg_why * 1.15:
                    _why_bullets.append(
                        f"🔥 {_wt} in form: {_wf.goals_scored_avg:.1f}g/game (last {_wf.n_games})"
                    )
                elif _wf.goals_scored_avg < _t_avg_why * 0.80:
                    _why_bullets.append(
                        f"🧊 {_wt} low output: {_wf.goals_scored_avg:.1f}g/game (last {_wf.n_games})"
                    )

        # 3. Tournament motivation (only when multiplier moves λ by ≥ 5%)
        if not _match_motivation.is_trivial():
            _STATUS_HE = {
                "must_win":               "חייב לנצח",
                "need_draw":              "צריך תיקו",
                "qualified_secure_1st":   "מובטח ראשון — עשוי לנוח שחקנים",
                "qualified_top_seed_fight": "נאבק על גרעין ראשון — הרכב מלא",
                "qualified":              "מוסמך — עשוי להחליף שחקנים",
                "eliminated":             "מודח — מוריד עצימות",
            }
            for _mt, _mm in (
                (match.home_team, _match_motivation.home),
                (match.away_team, _match_motivation.away),
            ):
                if abs(_mm.lambda_multiplier - 1.0) >= 0.05:
                    _ms = _STATUS_HE.get(_mm.qualification_status, _mm.qualification_status)
                    _why_bullets.append(f"🎯 {_mt}: {_ms} (עצימות ×{_mm.lambda_multiplier:.2f})")

        # 4. Winner market value (shown when model sees ≥ 5% edge over bookmaker)
        _edge_data = [
            (edge_h, sim.p_home, true_probs.home, odds_1x2.home, f"ניצחון {match.home_team}"),
            (edge_d, sim.p_draw, true_probs.draw, odds_1x2.draw, "תיקו"),
            (edge_a, sim.p_away, true_probs.away, odds_1x2.away, f"ניצחון {match.away_team}"),
        ]
        _top_edge = max(_edge_data, key=lambda x: x[0])
        if _top_edge[0] >= 0.05:
            _e, _bm, _bmkt, _bodd, _blbl = _top_edge
            _why_bullets.append(
                f"💡 VALUE: {_blbl} @ {_bodd:.2f}"
                f" — מודל {_bm:.0%} vs שוק {_bmkt:.0%} (יתרון {_e:+.0%})"
            )

        # 5. Goals expectation (Over/Under 2.5 from DC matrix + sum_goals value flag)
        if markets and markets.ou.get(2.5):
            _ou25  = markets.ou[2.5]
            _p_ov  = _ou25["p_over"]
            _p_un  = _ou25["p_under"]
            _exp_g = _ou25["expected_goals"]
            _ou_extra = f" | 🔥 VALUE: {sg_value_bet} גולים" if sg_value_bet else ""
            _why_bullets.append(
                f"⚽ צפי {_exp_g:.1f} גולים — Over 2.5: {_p_ov:.0%} | Under 2.5: {_p_un:.0%}{_ou_extra}"
            )

        # ── Opta supercomputer integration ───────────────────────────────────
        _opta_ctx  = build_opta_context(match.home_team, match.away_team)
        _opta_note = get_whatsapp_sentiment_note(
            match.home_team, match.away_team, sim.p_home, sim.p_away
        )
        _opta_tb   = opta_tiebreak(
            match.home_team, match.away_team,
            sim.p_home, sim.p_draw, sim.p_away,
        )
        if _opta_tb:
            _tb_ph, _tb_pd, _tb_pa = _opta_tb
            print(
                f"[opta] Tiebreak applied: p_home {sim.p_home:.1%}→{_tb_ph:.1%} "
                f"p_away {sim.p_away:.1%}→{_tb_pa:.1%}"
            )
        else:
            _tb_ph, _tb_pd, _tb_pa = sim.p_home, sim.p_draw, sim.p_away

        _tc_section = _match_motivation.to_ai_section()
        if _opta_ctx:
            _tc_section = _opta_ctx + ("\n\n" + _tc_section if _tc_section.strip() else "")

        ai_pick_prob = None
        ai_reasoning = None
        ensemble_pick = enhance(
            home_team                  = match.home_team,
            away_team                  = match.away_team,
            stage                      = stage,
            model                      = model,
            context_section            = _merged_context,
            value_bet_edge             = _active_edge,
            value_bet_outcome          = sim_value_bet or "",
            tournament_context_section = _tc_section,
        )
        if ensemble_pick:
            ai_pick_prob  = ensemble_pick.to_score_prob(model)
            # Prepend data source tag so WhatsApp shows provenance of AI analysis
            ai_reasoning  = f"[{_src_label}] {ensemble_pick.reasoning}"
        else:
            _offline = get_ai_offline_reason()
            if _offline:
                ai_reasoning = f"🤖 AI layer offline ({_offline}) — running on baseline Poisson"

        # Tournament context only relevant from matchday 3 onwards or in knockout rounds.
        # Rounds 1 & 2: all teams are motivated — suppress to keep notification clean.
        _is_knockout      = (stage != TournamentStage.GROUP_STAGE)
        _is_final_matchday = (
            _match_motivation.home.played >= 2 or
            _match_motivation.away.played >= 2
        )
        _show_context = _is_knockout or _is_final_matchday

        picks.append(DailyPick(
            home_team      = match.home_team,
            away_team      = match.away_team,
            recommendation = rec,
            ai_pick        = ai_pick_prob,
            ai_reasoning   = ai_reasoning,
            value_bets              = value_bets if value_bets else None,
            market_data             = markets,
            sg_value_bet            = sg_value_bet,
            tournament_context_lines = (
                (_match_motivation.to_whatsapp_lines() or []) +
                ([_opta_note] if _opta_note else []) +
                (["🔭 Opta tiebreak applied — simulation was indecisive"] if _opta_tb else [])
            ) or None if _show_context else (
                ([_opta_note] if _opta_note else None)
            ),
            logic_chain    = _logic_chain,
            why_bullets    = _why_bullets or None,
            # Simulation fields — drive final prediction in WhatsApp
            sim_score_home  = _sim_h,
            sim_score_away  = _sim_a,
            sim_p_home      = round(sim.p_home,          4),
            sim_p_draw      = round(sim.p_draw,          4),
            sim_p_away      = round(sim.p_away,          4),
            poisson_p_home  = round(sim.poisson_p_home,  4),
            poisson_p_draw  = round(sim.poisson_p_draw,  4),
            poisson_p_away  = round(sim.poisson_p_away,  4),
            lambda_home     = round(model.lambda_home,   3),
            lambda_away     = round(model.lambda_away,   3),
            is_knockout     = _is_knockout,
            sim_top3        = _sim_top3,
            pick_status     = _sim_pick_status,
        ))
        cs_picks.append(predict_correct_score(
            match.home_team, match.away_team, sim,
            external_xg=get_external_xg(match.home_team, match.away_team),
            is_knockout=_is_knockout,
        ))
        picks[-1].correct_score_pick = cs_picks[-1]
        morning_data.append({
            "date":             date.today().isoformat(),
            "home_team":        match.home_team,
            "away_team":        match.away_team,
            "stage":            stage.value,
            "lambda_home":       model.lambda_home,   # market-calibrated baseline (before adjustments)
            "lambda_away":       model.lambda_away,
            "final_lambda_home": lam_h,              # actual λ used for simulation (after all adjustments)
            "final_lambda_away": lam_a,
            "final_home_goals": _sim_h,   # simulation-derived score (was: AI pick or strategy advisor)
            "final_away_goals": _sim_a,
            "sim_p_home":       round(sim.p_home,       4),
            "sim_p_draw":       round(sim.p_draw,       4),
            "sim_p_away":       round(sim.p_away,       4),
            "sim_top3":         _sim_top3,
            "pick_status":      _sim_pick_status,
            "market_p_home":    round(true_probs.home,  4),
            "market_p_draw":    round(true_probs.draw,  4),
            "market_p_away":    round(true_probs.away,  4),
            "sim_value_bet":  sim_value_bet,
            "model_sg_01":    markets.sum_goals.get("0-1") if markets else None,
            "model_sg_23":    markets.sum_goals.get("2-3") if markets else None,
            "model_sg_4plus": markets.sum_goals.get("+4")  if markets else None,
            "home_motivation":        _match_motivation.home.qualification_status,
            "away_motivation":        _match_motivation.away.qualification_status,
            "home_lambda_multiplier": _match_motivation.home.lambda_multiplier,
            "away_lambda_multiplier": _match_motivation.away.lambda_multiplier,
            "kelly_value_bet":        _kvb.outcome      if _kvb else None,
            "kelly_value_bet_odds":   round(_kvb.decimal_odds, 3) if _kvb else None,
            "kelly_value_bet_stake":  round(_kvb_stake, 2) if _kvb_stake else None,
            "variance_mode": (_sim_h != _modal_h or _sim_a != _modal_a),
            "is_knockout":   _is_knockout,
            "predicted_by": "poisson_only",   # AI is audit-only — does not change the pick
        })

    if not picks:
        msg = "[pipeline] No matches could be analysed today (odds/schedule mismatch or all finished)."
        print(msg)
        if send_notification:
            send_whatsapp_message(msg)
        return msg

    # ── EV enrichment from winner_odds.json (no-op if file absent) ──────────
    morning_data = enrich_picks(morning_data, odds_path=_WINNER_ODDS_PATH)

    # ── Build EV-based value ticket (Value > 1.05 across all markets) ────────
    _all_value_legs: list[tuple[str, BetAnalysis]] = [
        (f"{pick.home_team} vs {pick.away_team}", vb)
        for pick in picks
        if pick.value_bets
        for vb in pick.value_bets
        if vb.is_value
    ]
    _ticket = build_ticket(_all_value_legs, bankroll=TOTAL_BANKROLL) if len(_all_value_legs) >= 2 else None
    if _ticket:
        print(
            f"[ticket/value] {len(_ticket.legs)}-leg value ticket: "
            f"odds={_ticket.combined_odds:.2f}  prob={_ticket.combined_prob:.1%}  "
            f"EV={_ticket.ev_combined:+.1%}  stake={_ticket.stake_nis:.0f} NIS"
        )

    # ── Build probability-maximizing straight-win ticket (Sim > 65%) ─────────
    _prob_candidates: list[tuple[str, str, str, float, float]] = []
    for _pk in picks:
        if not _pk.value_bets:
            continue
        _sim_h = _pk.sim_p_home or 0.0
        _sim_a = _pk.sim_p_away or 0.0
        _label = f"{_pk.home_team} vs {_pk.away_team}"
        # Choose whichever side (home or away) has the higher win probability
        if _sim_h >= _sim_a:
            _ba = next((b for b in _pk.value_bets if b.outcome == "Home Win"), None)
            if _ba:
                _prob_candidates.append((_label, "Home Win", _pk.home_team, _sim_h, _ba.decimal_odds))
        else:
            _ba = next((b for b in _pk.value_bets if b.outcome == "Away Win"), None)
            if _ba:
                _prob_candidates.append((_label, "Away Win", _pk.away_team, _sim_a, _ba.decimal_odds))

    _prob_ticket = build_probability_ticket(_prob_candidates, bankroll=TOTAL_BANKROLL)
    if _prob_ticket:
        print(
            f"[ticket/prob] {len(_prob_ticket.legs)}-leg probability ticket: "
            f"odds={_prob_ticket.combined_odds:.2f}  prob={_prob_ticket.combined_prob:.1%}  "
            f"stake={_prob_ticket.stake_nis:.0f} NIS"
        )

    # ── Build Confidence Value ticket (Sim ≥ 60% + edge ≥ 5% vs market) ─────
    # Uses the same _prob_candidates list (Home/Away Win only) but with lower
    # sim floor (60% vs 65%) so the edge filter does the heavy lifting.
    _conf_candidates = [
        (label, outcome, winner, sim_prob, dec_odds)
        for label, outcome, winner, sim_prob, dec_odds in _prob_candidates
    ]
    _conf_ticket = build_confidence_value_ticket(_conf_candidates, bankroll=TOTAL_BANKROLL)
    if _conf_ticket:
        print(
            f"[ticket/conf] {len(_conf_ticket.legs)}-leg confidence-value ticket: "
            f"odds={_conf_ticket.combined_odds:.2f}  prob={_conf_ticket.combined_prob:.1%}  "
            f"EV={_conf_ticket.total_ev:+.1%}  stake={_conf_ticket.stake_nis:.0f} NIS"
        )
    else:
        print("[ticket/conf] No Confidence Value bets found (Sim ≥ 60% + edge ≥ 5% on Winner).")

    # ── Step 5: Format + send ────────────────────────────────────────────────
    message = format_daily_message(
        picks, context, perf_report=perf_report,
        ticket=_ticket, prob_ticket=_prob_ticket, conf_ticket=_conf_ticket,
    )

    # no_odds_matches now receive prior-only predictions (included in picks above).
    # Only list here if they somehow never made it into picks (safety net).
    _prior_covered = {f"{p.home_team}|{p.away_team}" for p in picks}
    _truly_missing = [
        m for m in no_odds_matches
        if f"{m.home_team}|{m.away_team}" not in _prior_covered
    ]
    if _truly_missing:
        _no_odds_lines = ["\n\n📅 *מתוכנן — אין נתונים:*"]
        for _nm in _truly_missing:
            _ko_str = _nm.start_time_utc.strftime("%H:%M UTC")
            _no_odds_lines.append(f"  ⏳ {_nm.home_team} vs {_nm.away_team}  [{_ko_str}]")
        message += "\n".join(_no_odds_lines)

    # ── Submission validation gate ────────────────────────────────────────────
    # Re-simulate every pick from its final lambdas and assert the stored score
    # equals the Poisson modal.  Any divergence is logged loudly and appended to
    # the WhatsApp message so it is impossible to miss.
    _validation_errors = _validate_morning_picks(morning_data)
    if _validation_errors:
        print(f"[VALIDATION] WARNING — {len(_validation_errors)} pick(s) diverge from Poisson modal:")
        for _ve in _validation_errors:
            print(f"[VALIDATION]{_ve}")
        message += (
            "\n\n*[VALIDATION ALERT]* submitted score != model modal:\n" +
            "\n".join(_validation_errors) +
            "\nCheck pipeline logs — override was applied or a bug crept back in."
        )
    else:
        print(f"[VALIDATION] All {len(morning_data)} pick(s) match Poisson modal — pipeline clean.")

    if send_notification:
        send_whatsapp_message(message)
    else:
        print("\n--- Message (notification disabled) ---")
        print(message)
        print("--- End of message ---\n")

    if not dry_run:
        save_morning_picks(morning_data)
        _save_last_run(picks_generated=len(picks))
        # Persist learned team posteriors so the next run starts with
        # tournament-informed priors rather than static FIFA estimates.
        save_wc_priors(combined_results, _WC_PRIORS_PATH)
    return message


# ---------------------------------------------------------------------------
# Submission validation gate
# ---------------------------------------------------------------------------

def _validate_morning_picks(morning_data: list[dict]) -> list[str]:
    """
    Re-simulate each pick from final_lambda_home/away and verify the stored
    final_home_goals/final_away_goals matches the Poisson modal.

    Returns a list of human-readable divergence strings.  Empty list = all clear.

    This catches any code regression where a submitted score silently diverges
    from the model output — the failure mode that produced 3 wrong picks on
    2026-06-30 before the KO-boost and Math-First overrides were removed.
    """
    errors: list[str] = []
    for rec in morning_data:
        # EV-override intentionally deviates from modal — not a validation error
        if str(rec.get("pick_status", "")).startswith("override"):
            continue
        lh = rec.get("final_lambda_home") or rec.get("lambda_home")
        la = rec.get("final_lambda_away") or rec.get("lambda_away")
        if lh is None or la is None:
            continue
        modal_h, modal_a = simulate(lh, la).score_grid.most_likely_score()
        stored_h = rec.get("final_home_goals")
        stored_a = rec.get("final_away_goals")
        if (stored_h, stored_a) != (modal_h, modal_a):
            errors.append(
                f"  {rec['home_team']} vs {rec['away_team']}: "
                f"submitted {stored_h}-{stored_a} != modal {modal_h}-{modal_a} "
                f"(lam={lh:.3f}/{la:.3f})"
            )
    return errors


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

        # Rebuild Poisson model from saved lambda values (DC-corrected matrix)
        lh, la = record["lambda_home"], record["lambda_away"]
        model = PoissonMatchModel(
            lambda_home = lh,
            lambda_away = la,
            _matrix     = build_dc_matrix(lh, la),
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the single-run guard and re-run even if pipeline already ran today",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Skip all external API calls (odds, standings, AI, RapidAPI, notification). "
            "Step 0 (history ingestion + team-name matching) still runs so you can verify "
            "clean_team_name() without wasting quota. Implies --no-notify and --force. "
            "Does not write morning_picks.json or last_run.json."
        ),
    )
    args = parser.parse_args()

    if args.lineup_check:
        run_lineup_check_pipeline(send_notification=not args.no_notify)
    elif args.games_json:
        with open(args.games_json, encoding="utf-8") as f:
            raw_games = json.load(f)
        run_daily_pipeline(
            raw_games,
            send_notification=not args.no_notify and not args.dry_run,
            force=args.force or args.dry_run,   # dry-run always bypasses last_run guard
            dry_run=args.dry_run,
        )
    else:
        parser.error("games_json is required for the morning run (or use --lineup-check)")


if __name__ == "__main__":
    main()
