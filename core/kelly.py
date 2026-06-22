"""
core/kelly.py — Kelly Criterion + Value Bet analysis

Compares FDR-adjusted Poisson probabilities (our "true" edge) against raw
bookmaker decimal odds to identify value bets and optimal bet sizing.

Key formulas:
  implied_prob   = 1 / decimal_odds          (bookmaker's embedded probability)
  edge_pct       = our_prob / implied_prob - 1 (as a percentage)
  ev_per_unit    = our_prob * decimal_odds - 1
  kelly_fraction = ev_per_unit / (decimal_odds - 1)   (fraction of bankroll)
  half_kelly     = kelly_fraction / 2                  (recommended, safer)

Notes:
  - `our_prob` should be the FDR-modified Poisson probability, NOT derived
    from the same market odds (that would be circular).
  - Raw bookmaker `decimal_odds` are used (NOT overround-removed), because
    we are comparing our independent signal against what the market actually pays.
  - Kelly is capped at MAX_KELLY_FRACTION to guard against model overconfidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.poisson_engine import PoissonMatchModel
from core.odds_converter import MatchOdds1X2

# ── Tunable constants ────────────────────────────────────────────────────────

VALUE_BET_THRESHOLD  = 0.10   # minimum edge to flag as a value bet (10%)
MAX_KELLY_FRACTION   = 0.25   # cap Kelly at 25% of bankroll

# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class BetAnalysis:
    outcome:        str    # "Home Win" | "Draw" | "Away Win"
    our_prob:       float  # FDR-adjusted Poisson probability (0–1)
    decimal_odds:   float  # raw bookmaker decimal (e.g. 1.65)
    implied_prob:   float  # 1 / decimal_odds
    edge_pct:       float  # (our_prob / implied_prob - 1) * 100
    ev_per_unit:    float  # our_prob * decimal_odds - 1
    kelly_fraction: float  # full Kelly (capped at MAX_KELLY_FRACTION), 0 if no edge
    half_kelly:     float  # kelly_fraction / 2 (recommended bet size)
    is_value:       bool   # True when edge_pct >= VALUE_BET_THRESHOLD * 100


# ── Internal helpers ─────────────────────────────────────────────────────────

def _single_kelly(our_prob: float, decimal_odds: float) -> float:
    """
    Full Kelly fraction, capped at MAX_KELLY_FRACTION.
    Returns 0.0 when there is no positive edge (ev_per_unit <= 0).
    """
    ev = our_prob * decimal_odds - 1.0
    if ev <= 0:
        return 0.0
    net_odds = decimal_odds - 1.0
    if net_odds <= 0:
        return 0.0
    return min(ev / net_odds, MAX_KELLY_FRACTION)


def _analyse_outcome(
    outcome:      str,
    our_prob:     float,
    decimal_odds: float,
    threshold:    float,
) -> BetAnalysis:
    implied_prob   = 1.0 / decimal_odds if decimal_odds > 0 else 1.0
    edge_pct       = (our_prob / implied_prob - 1.0) * 100.0 if implied_prob > 0 else 0.0
    ev_per_unit    = our_prob * decimal_odds - 1.0
    kelly_fraction = _single_kelly(our_prob, decimal_odds)
    return BetAnalysis(
        outcome        = outcome,
        our_prob       = our_prob,
        decimal_odds   = decimal_odds,
        implied_prob   = implied_prob,
        edge_pct       = edge_pct,
        ev_per_unit    = ev_per_unit,
        kelly_fraction = kelly_fraction,
        half_kelly     = kelly_fraction / 2.0,
        is_value       = edge_pct >= threshold * 100.0,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def analyse_match(
    model:     PoissonMatchModel,
    raw_odds:  MatchOdds1X2,
    threshold: float = VALUE_BET_THRESHOLD,
) -> list[BetAnalysis]:
    """
    Run Kelly/EV analysis for all three 1X2 outcomes of a single match.

    Args:
        model:     FDR-adjusted PoissonMatchModel (provides our_prob for each outcome)
        raw_odds:  Bookmaker decimal odds BEFORE overround removal
        threshold: Minimum edge to flag as is_value (default 10%)

    Returns:
        List of three BetAnalysis objects: [Home Win, Draw, Away Win]
    """
    return [
        _analyse_outcome("Home Win", model.p_home_win(), raw_odds.home, threshold),
        _analyse_outcome("Draw",     model.p_draw(),     raw_odds.draw, threshold),
        _analyse_outcome("Away Win", model.p_away_win(), raw_odds.away, threshold),
    ]
