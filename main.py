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

from config.scoring_rules import TournamentStage
from core.ai_ensemble import enhance
from data.context_fetcher import fetch_match_context
from data.backup_scraper import fetch_match_context_espn
from data.results_fetcher import fetch_yesterday_results
from data.performance_tracker import ingest_results, load_history, save_history, yesterday_stats, compute_stats
from core.bias_corrector import build_bias_corrector
from data.fdr_fetcher import fetch_fixture_mu, apply_fdr_modifier
from core.kelly import analyse_match as analyse_match_bets, BetAnalysis
from core.simulator import simulate
from core.strength_model import build_strength_model, MIN_BLEND, BLEND_WEIGHT, _norm as _sm_norm
from core.market_calculator import calculate_all_markets
from data.winner_odds_loader import enrich_picks, get_all_odds, find_match_odds
from data.motivation import load_group_tables, build_match_motivation
from data.stats_collector import build_form_cache, FORM_BLEND_WEIGHT

_DATA_DIR            = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_MORNING_PICKS_PATH  = os.path.join(_DATA_DIR, "morning_picks.json")
_LAST_RUN_PATH       = os.path.join(_DATA_DIR, "last_run.json")
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

    # Build strength model from completed WC matches (returns None if < MIN_MATCHES)
    strength_model = build_strength_model(combined_results)
    if strength_model:
        print(strength_model.summary())

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

    # ── Step 1: Live standings sync ──────────────────────────────────────────
    if dry_run:
        print("[pipeline] Dry run — skipping standings sync (using hardcoded state).")
    else:
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
        msg = "[pipeline] No odds found for today — nothing to analyse. Check THE_ODDS_API_KEY or no WC matches today."
        print(msg)
        if send_notification:
            send_whatsapp_message(msg)
        return msg

    # ── Step 4: Match odds to today's schedule ────────────────────────────────
    todays_matches = get_todays_matches(all_matches)
    print(f"[pipeline] ── TODAY'S MATCHES FROM SCHEDULE ({len(todays_matches)} found) ──")
    for _i, _m in enumerate(todays_matches, 1):
        print(f"[pipeline]   {_i}. {_m.home_team} vs {_m.away_team}  "
              f"[{_m.start_time_utc.strftime('%Y-%m-%d %H:%M UTC')}]  status={_m.status}")
    print(f"[pipeline] ── END MATCH LIST ──────────────────────────────────────")

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
            lam_h = round((1 - BLEND_WEIGHT) * model.lambda_home + BLEND_WEIGHT * str_lh, 3)
            lam_a = round((1 - BLEND_WEIGHT) * model.lambda_away + BLEND_WEIGHT * str_la, 3)
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

        # ── Tournament Motivation (rotation-trap λ adjustment) ──────────────
        _match_motivation = build_match_motivation(
            match.home_team, match.away_team, _group_tables
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

        # ── Knockout stage intensity boost ───────────────────────────────────
        # Knockout matches tend toward higher-intensity, end-to-end play.
        # Group motivation system returns 'unknown' (×1.0) for knockout teams
        # since they are no longer tracked in group_tables.json — apply a fixed
        # 1.20× boost to both λ values to compensate for that gap.
        if stage != TournamentStage.GROUP_STAGE:
            _ko_boost = 1.20
            lam_h = round(lam_h * _ko_boost, 3)
            lam_a = round(lam_a * _ko_boost, 3)
            print(
                f"[motivation] Knockout stage ({stage.value}) intensity ×{_ko_boost:.2f}: "
                f"lam_h={lam_h}  lam_a={lam_a}"
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
        if stage != TournamentStage.GROUP_STAGE:
            _chain_steps.append("KO×1.20")
        _chain_steps.append(f"→ {lam_h:.2f}/{lam_a:.2f}")

        _total_h_pct = (_pct(_lam_h_market, lam_h) or "+0%")
        _logic_chain = " | ".join(_chain_steps) + f"  [{_total_h_pct} total vs market]"
        print(f"[chain] {match.home_team} vs {match.away_team}: {_logic_chain}")

        # ── Monte Carlo Simulation ──────────────────────────────────────────
        sim = simulate(lam_h, lam_a)
        print(
            f"[sim] Monte Carlo (n={sim.n_sims:,}): "
            f"H={sim.p_home:.1%}  D={sim.p_draw:.1%}  A={sim.p_away:.1%}"
        )
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

        # ── Kelly / Value Bet analysis ──────────────────────────────────────
        kelly_analyses = analyse_match_bets(model, odds_1x2)
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

        # P1: RapidAPI — injuries, confirmed form, lineups
        rapidapi_key    = os.environ.get("RAPIDAPI_KEY", "")
        match_ctx       = fetch_match_context(match.home_team, match.away_team, api_key=rapidapi_key)
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
            tournament_context_section = _match_motivation.to_ai_section(),
        )
        if ensemble_pick:
            ai_pick_prob  = ensemble_pick.to_score_prob(model)
            # Prepend data source tag so WhatsApp shows provenance of AI analysis
            ai_reasoning  = f"[{_src_label}] {ensemble_pick.reasoning}"

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
            tournament_context_lines = (_match_motivation.to_whatsapp_lines() or None) if _show_context else None,
            logic_chain    = _logic_chain,
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
            "sim_p_home":       round(sim.p_home,       4),
            "sim_p_draw":       round(sim.p_draw,       4),
            "sim_p_away":       round(sim.p_away,       4),
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
        })

    if not picks:
        msg = "[pipeline] No matches could be analysed today (odds/schedule mismatch or all finished)."
        print(msg)
        if send_notification:
            send_whatsapp_message(msg)
        return msg

    # ── EV enrichment from winner_odds.json (no-op if file absent) ──────────
    morning_data = enrich_picks(morning_data, odds_path=_WINNER_ODDS_PATH)

    # ── Step 5: Format + send ────────────────────────────────────────────────
    message = format_daily_message(picks, context, perf_report=perf_report)

    if send_notification:
        send_whatsapp_message(message)
    else:
        print("\n--- Message (notification disabled) ---")
        print(message)
        print("--- End of message ---\n")

    if not dry_run:
        save_morning_picks(morning_data)
        _save_last_run(picks_generated=len(picks))
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
