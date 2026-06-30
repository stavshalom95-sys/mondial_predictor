"""
core/correct_score_predictor.py — Dual-track prediction engine.

Professional Bet track (Strategy):
  Determines betting strategy — Safe Bet / Reduced Stake / Stay Away — by
  cross-referencing our internal Poisson model against optional external xG data.
  Strategy is ALWAYS based on 90-minute full-time probabilities (bookmaker standard).

Friends League track (Correct Score — Track A):
  Computes the most probable correct score using our Poisson score grid, blended
  40/60 with external xG lambdas when available.
  Track A = Poisson modal score (highest P(exact score)).  No direction override.

External xG data source:
  data/external_xg.json — manually populated from xG grid images before each run.
  If the file is absent or a match has no entry, falls back to internal Poisson only.

Strategy rules:
  • External draw% >= 0.33  → Stay Away (coin flip, no edge)
  • Max win prob  <  0.45   → Stay Away (too balanced)
  • External draw% >= 0.25  → Reduced Stake
  • External O2.5  <  0.35  → Reduced Stake (low-scoring risk) + Under 2.5 signal
  • |our_fav - ext_fav| > 0.10 → Prior Inflation flag → Reduced Stake

Measurement-First protocol (June 2026):
  - Math-First direction override REMOVED (Poisson modal score stands as-is)
  - ET simulation REMOVED (no calibration data to validate it)
  - Contrarian competition strategy SUSPENDED
"""
from __future__ import annotations

import json
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
    score_prob:      float   # P(this exact score) from Poisson modal
    score_label:     str     # "France wins 2-0 — narrow win"
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
    is_knockout:     bool = False   # kept for notifier compatibility; ET sim removed


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


def _score_label(home_team: str, away_team: str, h: int, a: int) -> str:
    """Concise score description. No heuristic qualifiers (low-scoring, cagey, etc.)."""
    if h == a:
        return f"Draw {h}-{a}"
    winner = home_team if h > a else away_team
    margin = abs(h - a)
    if margin == 1:
        tone = "narrow win"
    elif margin >= 3:
        tone = "dominant"
    else:
        tone = "comfortable win"
    return f"{winner} wins {max(h, a)}-{min(h, a)} — {tone}"


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
    is_knockout:  bool = False,   # kept for call-site compatibility; ET sim removed
) -> CorrectScorePick:
    """
    Produce a dual-track CorrectScorePick for one match.

    Track A — Friends League (Competition):
      Poisson modal score — highest P(exact score) from score grid.
      No direction override. No ET simulation.

    Track B — Betting (Strategy):
      90-min full-time probabilities, unchanged regardless of stage.

    When external_xg is provided, lambdas are blended 40%/60% and prior
    inflation is flagged when our model exceeds external by > 10pp.
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

    # ── Track A: Poisson modal score — highest P(exact score), no overrides ─────
    sh, sa, sp = score_grid.top_scores(1)[0]

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
        score_label     = _score_label(home_team, away_team, sh, sa),
        confidence      = confidence,
        strategy        = strategy,
        strategy_note   = strategy_note,
        kelly_cap       = kelly_cap,
        ou_signal       = ou_signal,
        prior_inflation = prior_inflation,
        source          = source,
        is_knockout     = is_knockout,
    )
