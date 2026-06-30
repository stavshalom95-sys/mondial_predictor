"""
core/correct_score_predictor.py — Dual-track prediction engine.

Professional Bet track (Strategy):
  Determines betting strategy — Safe Bet / Reduced Stake / Stay Away — by
  cross-referencing our internal Poisson model against optional external xG data.
  Strategy is ALWAYS based on 90-minute full-time probabilities (bookmaker standard).

Friends League track (Correct Score):
  Computes the most probable correct score using our Poisson score grid, blended
  40/60 with external xG lambdas when available.  When the external xG grid shows
  a clear modal score (e.g. 0-0 at 20%), that signal takes priority.

  Knockout matches: when the 90-min modal score is a draw, extra time (30 min) is
  simulated with a fatigue-reduced lambda scale (ET_LAMBDA_SCALE ≈ 0.25) to produce
  a decisive final result (AET goal or penalty winner).  The Friends League prediction
  therefore reflects the 120-min / full final result, not just 90 min.

External xG data source:
  data/external_xg.json — manually populated from xG grid images before each run.
  If the file is absent or a match has no entry, falls back to internal Poisson only.

Strategy rules (post-mortem June 2026):
  • External draw% >= 0.33  → Stay Away (coin flip, no edge)
  • Max win prob  <  0.45   → Stay Away (too balanced)
  • External draw% >= 0.25  → Reduced Stake
  • External O2.5  <  0.35  → Reduced Stake (low-scoring risk) + Under 2.5 signal
  • |our_fav - ext_fav| > 0.10 → Prior Inflation flag → Reduced Stake
  • External underdog xG >= 0.80 → draw resilience → Reduced Stake
"""
from __future__ import annotations

import json
import math
import os
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from core.simulator import simulate, SimResult

_EXTERNAL_XG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "external_xg.json"
)

# Blend weight: 40% external, 60% internal Poisson
EXTERNAL_BLEND = 0.40

# Strategy thresholds (90-min betting — unchanged)
DRAW_STAY_AWAY      = 0.33   # draw% >= this → Stay Away
DRAW_REDUCED_STAKE  = 0.25   # draw% >= this → Reduced Stake
OU_UNDER_GATE       = 0.35   # O2.5 < this  → Under 2.5 + Reduced Stake
PRIOR_INFLATION_GAP = 0.10   # our_fav - ext_fav > this → flag
UNDERDOG_XG_GATE    = 0.80   # underdog xG >= this → draw resilience signal

# Extra-time simulation constants (Friends League KO matches only)
# 30 min ET with fatigue + defensive caginess → effective rate ≈ 25% of 90-min λ
_ET_LAMBDA_SCALE = (30.0 / 90.0) * 0.75  # ≈ 0.25


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExternalXG:
    """One match entry from data/external_xg.json."""
    match_key:    str
    xg_home:      float   # external xG / lambda for home team
    xg_away:      float   # external xG / lambda for away team
    p_home:       float   # external 1X2 win probability
    p_draw:       float
    p_away:       float
    ou_over_2_5:  float   # P(total goals > 2.5)


@dataclass
class CorrectScorePick:
    """Dual-track output for one match."""
    home_team:       str
    away_team:       str
    # ── Friends League ──────────────────────────────────────────────────────
    score_home:      int
    score_away:      int
    score_prob:      float   # P(this exact score) — 90-min modal × ET branch prob for KO draws
    score_label:     str     # "France wins 2-0 AET — narrow win"
    confidence:      str     # "HIGH" | "MEDIUM" | "LOW"
    # ── Professional Bet ────────────────────────────────────────────────────
    strategy:        str     # "Safe Bet" | "Reduced Stake" | "Stay Away"
    strategy_note:   str     # one-line reason
    kelly_cap:       float   # 1.0 = full Kelly, 0.5 = half, 0.0 = no bet
    # ── Signals ─────────────────────────────────────────────────────────────
    ou_signal:       str     # "Over 2.5" | "Under 2.5" | "Neutral"
    prior_inflation: bool
    source:          str     # "blended" | "internal_only"
    # ── Format metadata ─────────────────────────────────────────────────────
    is_knockout:     bool = False   # True → Friends League pick accounts for 120 min / ET / pens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    name = name.strip()
    if name and not name[0].isascii():
        parts = name.split(None, 1)
        name = parts[1] if len(parts) > 1 else ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()


def _match_key(home: str, away: str) -> str:
    return f"{_norm(home)} vs {_norm(away)}"


def _score_label(
    home_team: str,
    away_team: str,
    h: int,
    a: int,
    is_low_scoring: bool,
    is_balanced: bool,
    et_suffix: str = "",      # "AET" | "pens" | "" (group stage)
    pen_winner: str = "",     # populated when et_suffix == "pens"
) -> str:
    # Knockout penalty shootout — score stays the same, winner decided on pens
    if et_suffix == "pens":
        return f"{pen_winner} wins {h}-{a} (pens) — penalty shootout"

    # Regular or AET decisive result
    if h == a:
        # Can only reach here for group-stage draws (et_suffix == "")
        if is_balanced:
            tone = "balanced stalemate"
        elif is_low_scoring:
            tone = "cagey draw"
        else:
            tone = "tight contest"
        return f"Draw {h}-{h} — {tone}"

    winner = home_team if h > a else away_team
    margin = abs(h - a)
    if margin == 1:
        tone = "narrow win"
    elif margin >= 3:
        tone = "dominant"
    else:
        tone = "comfortable win"

    aet_tag = " AET" if et_suffix == "AET" else ""
    return f"{winner} wins {max(h, a)}-{min(h, a)}{aet_tag} — {tone}"


# ---------------------------------------------------------------------------
# Extra-time simulation (knockout matches only)
# ---------------------------------------------------------------------------

def _et_outcome(
    lh: float,
    la: float,
    base_h: int,
    base_a: int,
) -> tuple[int, int, str, float]:
    """
    Simulate extra time (30 min) for a knockout match that ends level after 90 min.

    Uses Poisson P(score ≥ 1 in 30 min) with a fatigue/caginess scale factor
    (_ET_LAMBDA_SCALE ≈ 0.25).  Three possible ET outcomes:

      • Home only scores  → (base_h+1, base_a, "AET", p_home_only)
      • Away only scores  → (base_h, base_a+1, "AET", p_away_only)
      • Both or neither   → (base_h, base_a,   "pens", p_pens)

    Returns the most probable outcome as (final_h, final_a, suffix, branch_prob).
    """
    lh_et = lh * _ET_LAMBDA_SCALE
    la_et = la * _ET_LAMBDA_SCALE
    p_h = 1.0 - math.exp(-lh_et)   # P(home scores ≥1 in ET)
    p_a = 1.0 - math.exp(-la_et)   # P(away scores ≥1 in ET)

    p_home_only = p_h * (1.0 - p_a)
    p_away_only = (1.0 - p_h) * p_a
    p_pens      = 1.0 - p_home_only - p_away_only  # both score or neither

    options = [
        (base_h + 1, base_a,     "AET",  p_home_only),
        (base_h,     base_a + 1, "AET",  p_away_only),
        (base_h,     base_a,     "pens", p_pens),
    ]
    return max(options, key=lambda x: x[3])


# ---------------------------------------------------------------------------
# External xG loader
# ---------------------------------------------------------------------------

_xg_cache: Optional[dict[str, ExternalXG]] = None


def load_external_xg(path: str = _EXTERNAL_XG_PATH) -> dict[str, ExternalXG]:
    """
    Load data/external_xg.json into a normalised lookup dict.
    Returns empty dict if the file is absent or malformed.
    Result is cached for the process lifetime.
    """
    global _xg_cache
    if _xg_cache is not None:
        return _xg_cache

    _xg_cache = {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        count = 0
        for key, v in raw.items():
            if key.startswith("_"):
                continue
            parts = key.split(" vs ", 1)
            if len(parts) != 2:
                continue
            nk = _match_key(parts[0], parts[1])
            _xg_cache[nk] = ExternalXG(
                match_key   = nk,
                xg_home     = float(v.get("xg_home",     1.30)),
                xg_away     = float(v.get("xg_away",     1.10)),
                p_home      = float(v.get("p_home",      0.33)),
                p_draw      = float(v.get("p_draw",      0.33)),
                p_away      = float(v.get("p_away",      0.34)),
                ou_over_2_5 = float(v.get("ou_over_2_5", 0.50)),
            )
            count += 1
        if count:
            print(f"[xg] Loaded external xG for {count} match(es)")
    except FileNotFoundError:
        print("[xg] data/external_xg.json not found — internal Poisson only")
    except Exception as exc:
        print(f"[xg] Failed to load external_xg.json: {exc}")

    return _xg_cache


def get_external_xg(home: str, away: str) -> Optional[ExternalXG]:
    """Return ExternalXG for a match, or None if not in the cache."""
    return load_external_xg().get(_match_key(home, away))


# ---------------------------------------------------------------------------
# Strategy logic
# ---------------------------------------------------------------------------

def _determine_strategy(
    p_home: float,
    p_draw: float,
    p_away: float,
    ou_over: Optional[float],
    prior_inflation: bool,
    home_team: str,
    away_team: str,
    ext: Optional[ExternalXG],
) -> tuple[str, str, float]:
    """Return (strategy, note, kelly_cap)."""

    fav_prob = max(p_home, p_away)
    fav_name = home_team if p_home >= p_away else away_team

    # ── Stay Away gates ───────────────────────────────────────────────────────
    if p_draw >= DRAW_STAY_AWAY:
        return (
            "Stay Away",
            f"Draw {p_draw:.0%} — coin flip, no edge on winner",
            0.0,
        )
    if fav_prob < 0.45:
        return (
            "Stay Away",
            f"No clear favourite (best prob {fav_prob:.0%})",
            0.0,
        )

    # ── Reduced Stake gates ───────────────────────────────────────────────────
    reasons: list[str] = []
    if p_draw >= DRAW_REDUCED_STAKE:
        reasons.append(f"draw risk {p_draw:.0%}")
    if ou_over is not None and ou_over < OU_UNDER_GATE:
        reasons.append(f"O2.5 only {ou_over:.0%}")
    if prior_inflation:
        reasons.append("prior inflation vs external xG")
    if ext is not None:
        underdog_xg = min(ext.xg_home, ext.xg_away)
        if underdog_xg >= UNDERDOG_XG_GATE:
            reasons.append(f"underdog xG {underdog_xg:.2f} — draw resilience")

    if reasons:
        return (
            "Reduced Stake",
            "Reduced Stake — " + "; ".join(reasons),
            0.5,
        )

    # ── Safe Bet ──────────────────────────────────────────────────────────────
    return (
        "Safe Bet",
        f"Safe Bet — {fav_name} ({fav_prob:.0%} win probability)",
        1.0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(
    home_team:    str,
    away_team:    str,
    internal_sim: SimResult,
    external_xg:  Optional[ExternalXG] = None,
    is_knockout:  bool = False,
) -> CorrectScorePick:
    """
    Produce a dual-track CorrectScorePick for one match.

    When external_xg is provided:
      - Score grid is re-simulated with blended lambdas (40% ext, 60% internal)
      - Strategy probabilities come from the external 1X2
      - Prior inflation is flagged when our model > external by > 10pp

    When external_xg is None:
      - Score grid from internal simulation
      - Strategy from internal 1X2
      - Prior inflation check skipped

    When is_knockout is True (R16/QF/SF/Final):
      - Betting strategy is UNCHANGED — still 90-min full-time probabilities
      - Friends League score: if 90-min modal is a draw, extra time is simulated
        (30 min, λ scaled by _ET_LAMBDA_SCALE ≈ 0.25) to produce a decisive result.
        If ET also ends level, the higher-λ team is predicted to win on penalties.
    """
    lh_int = internal_sim.score_grid.lambda_home
    la_int = internal_sim.score_grid.lambda_away

    # ── Score grid ────────────────────────────────────────────────────────────
    if external_xg is not None:
        lh_eff = round(EXTERNAL_BLEND * external_xg.xg_home + (1 - EXTERNAL_BLEND) * lh_int, 3)
        la_eff = round(EXTERNAL_BLEND * external_xg.xg_away + (1 - EXTERNAL_BLEND) * la_int, 3)
        score_grid = simulate(lh_eff, la_eff).score_grid
        p_home  = external_xg.p_home
        p_draw  = external_xg.p_draw
        p_away  = external_xg.p_away
        ou_over = external_xg.ou_over_2_5
        source  = "blended"
    else:
        lh_eff  = lh_int
        la_eff  = la_int
        score_grid = internal_sim.score_grid
        p_home  = internal_sim.p_home
        p_draw  = internal_sim.p_draw
        p_away  = internal_sim.p_away
        ou_over = None
        source  = "internal_only"

    # ── Modal 90-min score ────────────────────────────────────────────────────
    sh, sa, sp = score_grid.top_scores(1)[0]

    is_low_scoring = (lh_int + la_int) < 2.0 or (ou_over is not None and ou_over < 0.42)
    is_balanced    = abs(p_home - p_away) < 0.10

    # ── Extra-time resolution (Friends League KO only) ────────────────────────
    # Betting strategy always uses 90-min probabilities — no change below.
    et_suffix  = ""
    pen_winner = ""
    if is_knockout and sh == sa:
        et_h, et_a, et_suffix, et_branch_prob = _et_outcome(lh_eff, la_eff, sh, sa)
        sp = round(sp * et_branch_prob, 4)   # P(FT draw) × P(ET branch)
        sh, sa = et_h, et_a
        if et_suffix == "pens":
            pen_winner = home_team if lh_eff >= la_eff else away_team

    # ── Prior inflation ───────────────────────────────────────────────────────
    prior_inflation = False
    if external_xg is not None:
        ext_fav = max(external_xg.p_home, external_xg.p_away)
        our_fav = max(internal_sim.p_home, internal_sim.p_away)
        prior_inflation = (our_fav - ext_fav) > PRIOR_INFLATION_GAP

    # ── O/U signal ────────────────────────────────────────────────────────────
    if ou_over is None:
        ou_signal = "Neutral"
    elif ou_over < OU_UNDER_GATE:
        ou_signal = "Under 2.5"
    elif ou_over > 0.60:
        ou_signal = "Over 2.5"
    else:
        ou_signal = "Neutral"

    # ── Strategy (90-min FT — bookmaker standard, never modified for KO) ──────
    strategy, strategy_note, kelly_cap = _determine_strategy(
        p_home, p_draw, p_away, ou_over, prior_inflation,
        home_team, away_team, external_xg,
    )

    # ── Confidence ────────────────────────────────────────────────────────────
    fav_prob = max(p_home, p_away)
    if strategy == "Stay Away":
        confidence = "LOW"
    elif strategy == "Reduced Stake":
        confidence = "MEDIUM"
    elif fav_prob > 0.62:
        confidence = "HIGH"
    else:
        confidence = "MEDIUM"

    return CorrectScorePick(
        home_team       = home_team,
        away_team       = away_team,
        score_home      = sh,
        score_away      = sa,
        score_prob      = round(sp, 4),
        score_label     = _score_label(
            home_team, away_team, sh, sa,
            is_low_scoring, is_balanced,
            et_suffix=et_suffix, pen_winner=pen_winner,
        ),
        confidence      = confidence,
        strategy        = strategy,
        strategy_note   = strategy_note,
        kelly_cap       = kelly_cap,
        ou_signal       = ou_signal,
        prior_inflation = prior_inflation,
        source          = source,
        is_knockout     = is_knockout,
    )
