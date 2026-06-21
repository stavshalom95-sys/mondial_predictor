"""
Convert raw bookmaker decimal odds to true probabilities by removing the overround.
Uses the Multiplicative (proportional) method.
"""
from dataclasses import dataclass


@dataclass
class MatchOdds1X2:
    home: float  # decimal odds, e.g. 1.65
    draw: float
    away: float


@dataclass
class OverUnderOdds:
    line: float   # e.g. 2.5
    over: float   # decimal odds
    under: float  # decimal odds


@dataclass
class TrueProbs1X2:
    home: float
    draw: float
    away: float
    overround: float  # raw implied probability sum - 1.0 (bookmaker margin)


@dataclass
class TrueProbsOU:
    p_over: float
    p_under: float
    line: float


def remove_overround(odds: MatchOdds1X2) -> TrueProbs1X2:
    """
    Multiplicative method: divide each implied prob by the total sum.
    Example: home=1.65, draw=4.00, away=5.50
      raw = [0.6061, 0.2500, 0.1818] -> total = 1.0379 (overround ~3.79%)
      true = [0.5840, 0.2409, 0.1752]
    """
    raw = [1.0 / odds.home, 1.0 / odds.draw, 1.0 / odds.away]
    total = sum(raw)
    overround = total - 1.0
    home, draw, away = (p / total for p in raw)
    return TrueProbs1X2(home=home, draw=draw, away=away, overround=overround)


def remove_overround_ou(odds: OverUnderOdds) -> TrueProbsOU:
    """Remove overround from an Over/Under market."""
    raw = [1.0 / odds.over, 1.0 / odds.under]
    total = sum(raw)
    p_over, p_under = (p / total for p in raw)
    return TrueProbsOU(p_over=p_over, p_under=p_under, line=odds.line)
