"""
performance_tracker.py — Persist and score prediction history.

Called each morning by run_daily_pipeline() to:
  1. Ingest yesterday's actual results against yesterday's predictions.
  2. Return a summary dict for the WhatsApp performance block.

History is stored in data/history.json as a list of records:
  {
    "date":              "2026-06-14",
    "home_team":         "Spain",
    "away_team":         "Saudi Arabia",
    "stage":             "שלב הבתים",
    "predicted_home":    3,
    "predicted_away":    0,
    "actual_home":       2,
    "actual_away":       0,
    "exact_match":       false,
    "correct_result":    true,
    "points_earned":     1,
    "points_possible":   3,
  }
"""
from __future__ import annotations

import json
import os
import unicodedata
from datetime import date, timedelta
from typing import Optional

from config.scoring_rules import SCORING, TournamentStage

_HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "history.json"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_history() -> list[dict]:
    """Load history.json; return empty list if file absent or corrupt."""
    if not os.path.exists(_HISTORY_PATH):
        return []
    try:
        with open(_HISTORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"[tracker] Warning: could not load history: {exc}")
        return []


def save_history(history: list[dict]) -> None:
    """Persist history list to history.json."""
    os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
    print(f"[tracker] save_history() called with {len(history)} record(s).")
    try:
        with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"[tracker] history.json written ({len(history)} records) → {_HISTORY_PATH}")
    except Exception as exc:
        print(f"[tracker] ERROR: could not write history.json: {exc}")


def ingest_results(
    morning_picks: list[dict],
    results: list[dict],
    existing_history: list[dict],
) -> list[dict]:
    """
    Merge yesterday's picks with actual results into existing_history.

    morning_picks — records from morning_picks.json (added "date" key).
    results       — output of fetch_yesterday_results().
    existing_history — current history.json contents.

    Returns updated history (existing + new records). Idempotent: already-
    scored records (matched by date+teams) are not duplicated.
    """
    # Build dedup set from existing records
    seen: set[tuple[str, str, str]] = {
        (_r["date"], _normalize(_r["home_team"]), _normalize(_r["away_team"]))
        for _r in existing_history
        if "date" in _r
    }

    print(f"[tracker] ingest_results(): {len(morning_picks)} pick(s), {len(results)} result(s), {len(existing_history)} existing record(s).")
    # Always dump the full result list so we can see exactly what the API returned
    if results:
        print(f"[tracker] All finished results available for matching:")
        for _r in results:
            print(f"[tracker]   '{_r['home_team']}' vs '{_r['away_team']}' "
                  f"({_r.get('home_goals', '?')}-{_r.get('away_goals', '?')})")
    else:
        print("[tracker] WARNING: results list is empty — no finished matches were passed in.")
    new_records: list[dict] = []

    for pick in morning_picks:
        pick_date = pick.get("date", "")
        home_team = pick.get("home_team", "")
        away_team = pick.get("away_team", "")

        dedup_key = (pick_date, _normalize(home_team), _normalize(away_team))
        if dedup_key in seen:
            print(f"[tracker]   Already scored: {home_team} vs {away_team} ({pick_date}) — skipping.")
            continue

        # Find corresponding actual result
        result = _find_result(home_team, away_team, results)
        if result is None:
            print(f"[tracker]   ✗ No match for pick '{home_team}' vs '{away_team}' "
                  f"(pick date: {pick_date}) — not in any finished result above.")
            continue

        actual_home = result["home_goals"]
        actual_away = result["away_goals"]
        pred_home   = pick["final_home_goals"]
        pred_away   = pick["final_away_goals"]

        try:
            stage = TournamentStage(pick.get("stage", TournamentStage.GROUP_STAGE.value))
        except ValueError:
            stage = TournamentStage.GROUP_STAGE

        pts_earned, pts_possible, exact, correct = _score_pick(
            pred_home, pred_away, actual_home, actual_away, stage
        )

        # ── Bet P&L (populated when kelly_value_bet info present in pick) ────
        kvb_outcome = pick.get("kelly_value_bet")       # "Home Win" | "Draw" | "Away Win"
        kvb_odds    = pick.get("kelly_value_bet_odds")
        kvb_stake   = pick.get("kelly_value_bet_stake")
        bet_won     = None
        pnl_nis     = None
        if kvb_outcome and kvb_odds and kvb_stake:
            _outcome_dir = {"Home Win": "1", "Draw": "X", "Away Win": "2"}
            actual_dir   = _direction(actual_home, actual_away)
            if kvb_outcome in _outcome_dir:
                bet_won = (actual_dir == _outcome_dir[kvb_outcome])
                pnl_nis = round(
                    (float(kvb_odds) - 1) * float(kvb_stake)
                    if bet_won else -float(kvb_stake),
                    2,
                )

        record = {
            "date":            pick_date,
            "home_team":       home_team,
            "away_team":       away_team,
            "stage":           stage.value,
            "predicted_home":  pred_home,
            "predicted_away":  pred_away,
            "actual_home":     actual_home,
            "actual_away":     actual_away,
            "exact_match":     exact,
            "correct_result":  correct,
            "points_earned":   pts_earned,
            "points_possible": pts_possible,
            "bet_won":         bet_won,
            "pnl_nis":         pnl_nis,
            "predicted_by":    pick.get("predicted_by", "unknown"),
        }
        new_records.append(record)
        seen.add(dedup_key)

        icon = "🎯" if exact else ("✅" if correct else "❌")
        print(
            f"[tracker]   {icon} {home_team} {pred_home}-{pred_away} "
            f"(predicted) vs {actual_home}-{actual_away} (actual) "
            f"→ {pts_earned}/{pts_possible} pts"
        )

    print(f"[tracker] Ingested {len(new_records)} new record(s).")
    return existing_history + new_records


def yesterday_stats(history: list[dict]) -> Optional[dict]:
    """
    Return performance summary for yesterday's matches, or None if no
    records exist for yesterday.

    Returns:
        {
          "date_label":   "20/06",        ← DD/MM for Hebrew message
          "correct":      3,              ← correct result (1X2)
          "total":        4,              ← matches with results
          "exact":        1,              ← exact score hits
          "pts_earned":   7,
          "pts_possible": 12,
        }
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    records = [r for r in history if r.get("date") == yesterday]
    if not records:
        return None

    stats = compute_stats(records)
    # Convert ISO date to DD/MM for Hebrew display
    y, m, d_ = yesterday.split("-")
    stats["date_label"] = f"{d_}/{m}"
    return stats


def compute_stats(records: list[dict]) -> dict:
    """Aggregate stats over an arbitrary set of records."""
    total        = len(records)
    correct      = sum(1 for r in records if r.get("correct_result"))
    exact        = sum(1 for r in records if r.get("exact_match"))
    pts_earned   = sum(r.get("points_earned", 0)   for r in records)
    pts_possible = sum(r.get("points_possible", 0) for r in records)
    # P&L: aggregate only records where a value bet was placed and result known
    bet_records  = [r for r in records if r.get("pnl_nis") is not None]
    pnl_total    = round(sum(r["pnl_nis"] for r in bet_records), 2) if bet_records else None

    # Brier score (proper scoring rule) — lower is better; random baseline = 0.333
    brier = brier_score(records)

    # Per-source win rate breakdown
    by_source: dict[str, dict] = {}
    for r in records:
        src = r.get("predicted_by", "unknown")
        if src not in by_source:
            by_source[src] = {"total": 0, "correct": 0}
        by_source[src]["total"] += 1
        if r.get("correct_result"):
            by_source[src]["correct"] += 1
    win_rate_by_source = {
        src: {
            "total":    v["total"],
            "correct":  v["correct"],
            "win_rate": round(v["correct"] / v["total"], 3) if v["total"] else None,
        }
        for src, v in by_source.items()
    }

    return {
        "total":               total,
        "correct":             correct,
        "exact":               exact,
        "pts_earned":          pts_earned,
        "pts_possible":        pts_possible,
        "pnl_nis":             pnl_total,
        "bets_placed":         len(bet_records),
        "brier_score":         brier,
        "win_rate_by_source":  win_rate_by_source,
    }


def brier_score(records: list[dict]) -> Optional[float]:
    """
    Multi-class Brier score for 1X2 outcome predictions.

    AI Research Skills: brainstorming-research-ideas Framework 4 (Cross-Pollination).
    Proper scoring rule from meteorological forecasting (Brier 1950), adopted
    by ML for probability calibration assessment.

    Score = (1/N) * Σ [ (p_home - y_home)² + (p_draw - y_draw)² + (p_away - y_away)² ]
    where y_* ∈ {0, 1} is the one-hot actual outcome.

    Range: 0 (perfect) to 2 (maximally wrong). Random baseline for 3-class = 0.667.

    Requires records to have poisson_p_home/poisson_p_draw/poisson_p_away fields.
    Returns None when no records have these stored probabilities.
    """
    scored_records = [
        r for r in records
        if r.get("poisson_p_home") and r.get("poisson_p_draw") and r.get("poisson_p_away")
        and r.get("actual_home") is not None and r.get("actual_away") is not None
    ]
    if not scored_records:
        return None

    total_bs = 0.0
    for rec in scored_records:
        ph = float(rec["poisson_p_home"])
        pd = float(rec["poisson_p_draw"])
        pa = float(rec["poisson_p_away"])

        ah = int(rec["actual_home"])
        aa = int(rec["actual_away"])

        # One-hot encode actual outcome
        y_home = 1.0 if ah > aa else 0.0
        y_draw = 1.0 if ah == aa else 0.0
        y_away = 1.0 if ah < aa else 0.0

        total_bs += (ph - y_home)**2 + (pd - y_draw)**2 + (pa - y_away)**2

    return round(total_bs / len(scored_records), 4)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def clean_team_name(name: str) -> str:
    """
    Remove emoji, flag characters, and non-letter symbols from a team name.

    Unicode categories kept: Letters (L*), Numbers (N*), Space separators (Zs),
    and Punctuation (P* — hyphens, apostrophes, etc.).

    Everything else is stripped:
      S* — Symbol categories (So = emoji/flags, Sc = currency, Sm = math, Sk = modifier)
      C* — Control/Format (Cf = ZWJ used in emoji sequences, Cc = control chars)
      M* — Combining marks (handled further by NFD in _normalize)

    Examples:
      '🇪🇸 Spain'             → 'Spain'
      '🏴\u200d☠️ Somalia'   → 'Somalia'
      'Côte d\'Ivoire'        → 'Côte d\'Ivoire'  (preserved for NFD in _normalize)
    """
    filtered = "".join(
        c for c in name
        if unicodedata.category(c).startswith(("L", "N", "Z", "P"))
    )
    return " ".join(filtered.split())


def _normalize(name: str) -> str:
    """Lowercase, strip, collapse whitespace, remove emoji and diacritics."""
    name = clean_team_name(name)            # strip emoji / flags / symbols first
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return " ".join(name.lower().split())


def _teams_match(a: str, b: str) -> bool:
    """True if two team names refer to the same team after normalization.

    Matching tiers (first match wins):
      1. Exact normalized strings       "south korea" == "south korea"
      2. Substring                      "iran" in "ir iran"
      3. Significant-word overlap       "korea republic" ∩ "south korea" → {"korea"}
    """
    na, nb = _normalize(a), _normalize(b)
    if na == nb or na in nb or nb in na:
        return True
    # Tier 3: any word >3 chars in common handles common API variant names:
    #   "Korea Republic" vs "South Korea", "Ivory Coast" vs "Cote d'Ivoire" (partial),
    #   "United States" vs "USA" will NOT match here (both <4 chars) — kept intentionally strict
    words_a = {w for w in na.split() if len(w) > 3}
    words_b = {w for w in nb.split() if len(w) > 3}
    return bool(words_a & words_b)


def _find_result(
    home_team: str,
    away_team: str,
    results: list[dict],
) -> Optional[dict]:
    """
    Find the result dict for (home_team, away_team) from fetch_yesterday_results().
    Tries exact normalized match first, then swapped (neutral-venue edge case).
    """
    for r in results:
        if _teams_match(home_team, r["home_team"]) and _teams_match(away_team, r["away_team"]):
            return r
    # Swapped order
    for r in results:
        if _teams_match(home_team, r["away_team"]) and _teams_match(away_team, r["home_team"]):
            # Return with goals swapped to match the pick's home/away orientation
            return {
                "home_goals": r["away_goals"],
                "away_goals": r["home_goals"],
            }
    return None


def _direction(home_goals: int, away_goals: int) -> str:
    """Return '1', 'X', or '2' for home win, draw, away win."""
    if home_goals > away_goals:
        return "1"
    if home_goals < away_goals:
        return "2"
    return "X"


def _score_pick(
    pred_home: int,
    pred_away: int,
    actual_home: int,
    actual_away: int,
    stage: TournamentStage,
) -> tuple[int, int, bool, bool]:
    """
    Score a single prediction.

    Returns (points_earned, points_possible, exact_match, correct_result).
    """
    rules        = SCORING.get(stage, SCORING[TournamentStage.GROUP_STAGE])
    pts_possible = rules["exact"]

    exact   = (pred_home == actual_home) and (pred_away == actual_away)
    correct = _direction(pred_home, pred_away) == _direction(actual_home, actual_away)

    if exact:
        pts_earned = rules["exact"]
    elif correct:
        pts_earned = rules["direction"]
    else:
        pts_earned = 0

    return pts_earned, pts_possible, exact, correct
