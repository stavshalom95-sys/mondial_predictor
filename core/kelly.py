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

from dataclasses import dataclass, field
from typing import Optional

from core.poisson_engine import PoissonMatchModel
from core.odds_converter import MatchOdds1X2

# ── Tunable constants ────────────────────────────────────────────────────────

VALUE_BET_THRESHOLD  = 0.05   # minimum edge: Value > 1.05 (model_prob × decimal_odds > 1.05)
MAX_KELLY_FRACTION   = 0.25   # cap Kelly at 25% of bankroll

# ── Data models ──────────────────────────────────────────────────────────────

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
    is_value:       bool   # True when our_prob * decimal_odds > 1 + VALUE_BET_THRESHOLD
    value:          float  = field(default=0.0)   # our_prob * decimal_odds (the "Value" score)


@dataclass
class TicketLeg:
    match_label:  str    # "Portugal vs Uzbekistan"
    outcome:      str    # "Home Win" | "Draw" | "Away Win"
    decimal_odds: float
    our_prob:     float
    value:        float  # our_prob * decimal_odds


@dataclass
class Ticket:
    legs:          list[TicketLeg]
    combined_odds: float   # product of all leg decimal odds
    combined_prob: float   # product of all leg model probs (independence assumption)
    ev_combined:   float   # combined_prob * combined_odds - 1
    kelly_frac:    float   # full Kelly for the parlay (capped)
    stake_nis:     float   # half-Kelly × bankroll (caller supplies bankroll)


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
    value_score    = our_prob * decimal_odds                          # the "Value" number
    edge_pct       = (value_score - 1.0) * 100.0                     # same as EV%
    ev_per_unit    = value_score - 1.0
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
        is_value       = value_score > 1.0 + threshold,
        value          = round(value_score, 4),
    )


def build_ticket(
    value_legs: list[tuple[str, "BetAnalysis"]],
    bankroll:   float,
    max_legs:   int = 3,
) -> Optional["Ticket"]:
    """
    Construct a Double or Triple from the top value legs (sorted by EV).

    Args:
        value_legs: list of (match_label, BetAnalysis) where is_value=True
        bankroll:   total bankroll in NIS for Kelly sizing
        max_legs:   maximum legs in ticket (default 3 = Triple)

    Returns:
        Ticket, or None when fewer than 2 legs or combined EV is negative.
    """
    if len(value_legs) < 2:
        return None

    # Rank legs by ev_per_unit descending; cap at max_legs
    top = sorted(value_legs, key=lambda x: x[1].ev_per_unit, reverse=True)[:max_legs]

    legs = [
        TicketLeg(
            match_label  = label,
            outcome      = ba.outcome,
            decimal_odds = ba.decimal_odds,
            our_prob     = ba.our_prob,
            value        = ba.value,
        )
        for label, ba in top
    ]

    combined_odds = 1.0
    combined_prob = 1.0
    for leg in legs:
        combined_odds *= leg.decimal_odds
        combined_prob *= leg.our_prob

    ev_combined = combined_prob * combined_odds - 1.0
    if ev_combined <= 0:
        return None

    net_combined  = combined_odds - 1.0
    kelly_frac    = min(ev_combined / net_combined, MAX_KELLY_FRACTION) if net_combined > 0 else 0.0
    stake_nis     = (kelly_frac / 2.0) * bankroll   # half-Kelly

    return Ticket(
        legs          = legs,
        combined_odds = round(combined_odds, 2),
        combined_prob = round(combined_prob, 4),
        ev_combined   = round(ev_combined,   4),
        kelly_frac    = round(kelly_frac,    4),
        stake_nis     = round(stake_nis,     1),
    )


def build_probability_ticket(
    candidates: list[tuple[str, str, str, float, float]],
    bankroll:   float,
    min_prob:   float = 0.65,
    max_legs:   int   = 3,
) -> Optional["Ticket"]:
    """
    Build a high-probability straight-win ticket (Double or Triple).

    Ignores Value/EV filter — goal is maximum combined win probability.
    Only Home Win or Away Win outcomes are accepted (no Draws).

    Args:
        candidates: list of (match_label, outcome, winner_name, sim_prob, decimal_odds)
                    outcome must be "Home Win" or "Away Win"
        bankroll:   total bankroll NIS for Kelly sizing
        min_prob:   minimum MC simulation win probability to qualify (default 65%)
        max_legs:   cap ticket at this many legs (default 3)

    Returns:
        Ticket with legs sorted by sim_prob descending, or None if < 2 qualify.
    """
    qualified = [c for c in candidates if c[3] >= min_prob]
    if len(qualified) < 2:
        return None

    # Sort by probability descending — maximize combined hit rate
    qualified.sort(key=lambda x: x[3], reverse=True)
    top = qualified[:max_legs]

    legs = [
        TicketLeg(
            match_label  = label,
            outcome      = outcome,
            decimal_odds = decimal_odds,
            our_prob     = sim_prob,
            value        = round(sim_prob * decimal_odds, 4),
        )
        for label, outcome, _winner, sim_prob, decimal_odds in top
    ]

    combined_odds = 1.0
    combined_prob = 1.0
    for leg in legs:
        combined_odds *= leg.decimal_odds
        combined_prob *= leg.our_prob

    ev_combined  = combined_prob * combined_odds - 1.0
    net_combined = combined_odds - 1.0
    kelly_frac   = (
        min(ev_combined / net_combined, MAX_KELLY_FRACTION)
        if ev_combined > 0 and net_combined > 0 else 0.0
    )
    stake_nis = (kelly_frac / 2.0) * bankroll

    return Ticket(
        legs          = legs,
        combined_odds = round(combined_odds, 2),
        combined_prob = round(combined_prob, 4),
        ev_combined   = round(ev_combined,   4),
        kelly_frac    = round(kelly_frac,    4),
        stake_nis     = round(stake_nis,     1),
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
