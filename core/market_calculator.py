"""
market_calculator.py — Sub-market probability calculators.

Derives Sum-Goals distribution, Goal Difference, Asian Handicap, and Both Teams
To Score (BTTS) probabilities from the Dixon-Coles corrected score matrix.

All public functions accept (lam_h, lam_a) — the same λ values produced by
calibrate_dc() or the strength-blended pipeline in main.py — so they slot in
with zero additional API calls or data sources.

Typical usage in main.py (after model is built and lambdas are set):

    from core.market_calculator import MarketResult, calculate_all_markets

    markets = calculate_all_markets(lam_h, lam_a, home_team, away_team)
    print(markets.summary())

    # Sum-goals distribution (3-way bracket):
    sg = markets.sum_goals
    print(f"Goals 0-1: {sg['0-1']:.1%}  2-3: {sg['2-3']:.1%}  4+: {sg['+4']:.1%}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.poisson_engine import build_dc_matrix, MAX_GOALS


# ---------------------------------------------------------------------------
# Over/Under Goals
# ---------------------------------------------------------------------------

def ou_probabilities(
    lam_h: float,
    lam_a: float,
    lines: Optional[list[float]] = None,
) -> dict[float, dict]:
    """
    Compute Over/Under probabilities for one or more total-goals lines.

    Uses the DC-corrected matrix so the low-score corrections (especially
    the 0-0 boost) propagate correctly into the under-2.5 probability.

    Args:
        lam_h:  Expected home goals.
        lam_a:  Expected away goals.
        lines:  Total-goals thresholds to evaluate (default: [1.5, 2.5, 3.5, 4.5]).

    Returns:
        Dict keyed by line value, each containing:
            p_over  : P(total goals > line)
            p_under : P(total goals < line)  [= 1 - p_over, no push for .5 lines]
            expected_goals : lam_h + lam_a

    Example:
        >>> ou = ou_probabilities(1.845, 1.244, [2.5, 3.5])
        >>> ou[2.5]
        {'p_over': 0.6123, 'p_under': 0.3877, 'expected_goals': 3.089}
    """
    if lines is None:
        lines = [1.5, 2.5, 3.5, 4.5]

    matrix = build_dc_matrix(lam_h, lam_a)
    n      = len(matrix)
    result: dict[float, dict] = {}

    for line in lines:
        threshold = int(line + 0.5)   # 2.5 → 3,  3.5 → 4,  1.5 → 2
        p_over = sum(
            matrix[h][a]
            for h in range(n)
            for a in range(n)
            if h + a >= threshold
        )
        result[line] = {
            "p_over":          round(p_over, 4),
            "p_under":         round(1.0 - p_over, 4),
            "expected_goals":  round(lam_h + lam_a, 3),
        }

    return result


# ---------------------------------------------------------------------------
# Sum-Goals Distribution (3-way bracket)
# ---------------------------------------------------------------------------

def sum_goals_distribution(lam_h: float, lam_a: float) -> dict[str, float]:
    """
    Three-bracket total-goals distribution derived from the DC-corrected matrix.

    Brackets:
        "0-1": P(total goals <= 1)
        "2-3": P(total goals == 2 or 3)
        "+4":  P(total goals >= 4)

    Useful for betting markets that offer a 3-way total-goals line rather
    than a binary Over/Under 2.5.

    Example:
        >>> sg = sum_goals_distribution(1.4, 0.9)
        >>> sg
        {'0-1': 0.2341, '2-3': 0.4512, '+4': 0.3147}
    """
    matrix = build_dc_matrix(lam_h, lam_a)
    n      = len(matrix)
    p_01 = sum(matrix[h][a] for h in range(n) for a in range(n) if h + a <= 1)
    p_23 = sum(matrix[h][a] for h in range(n) for a in range(n) if 2 <= h + a <= 3)
    p_4p = sum(matrix[h][a] for h in range(n) for a in range(n) if h + a >= 4)
    return {
        "0-1": round(p_01, 4),
        "2-3": round(p_23, 4),
        "+4":  round(p_4p, 4),
    }


# ---------------------------------------------------------------------------
# Goal Difference Distribution
# ---------------------------------------------------------------------------

def goal_diff_distribution(
    lam_h: float,
    lam_a: float,
) -> dict[int, float]:
    """
    Full probability distribution over goal differences (home − away).

    Positive GD = home team wins by that margin.
    Negative GD = away team wins by that margin.
    Zero        = draw.

    Derived by summing the DC matrix along each anti-diagonal.

    Args:
        lam_h:  Expected home goals.
        lam_a:  Expected away goals.

    Returns:
        Dict {goal_diff: probability} sorted from most negative to most positive.

    Example:
        >>> dist = goal_diff_distribution(1.845, 1.244)
        >>> dist[0]   # P(draw)
        0.2554
        >>> dist[1]   # P(home wins by exactly 1)
        0.1876
    """
    matrix = build_dc_matrix(lam_h, lam_a)
    n      = len(matrix)
    dist: dict[int, float] = {}

    for h in range(n):
        for a in range(n):
            gd = h - a
            dist[gd] = dist.get(gd, 0.0) + matrix[h][a]

    return {k: round(v, 4) for k, v in sorted(dist.items())}


# ---------------------------------------------------------------------------
# Asian Handicap
# ---------------------------------------------------------------------------

def asian_handicap(
    lam_h: float,
    lam_a: float,
    handicap: float,
) -> dict:
    """
    Probability that the home team covers an Asian Handicap line.

    Convention (standard bookmaker):
        handicap > 0 : home team RECEIVES goals   (e.g., +1.5 = home wins if lose by ≤1)
        handicap < 0 : home team GIVES goals       (e.g., -1.5 = home must win by ≥2)
        handicap = 0 : moneyline (push if draw)

    For half-ball handicaps (e.g., ±0.5, ±1.5, ±2.5):
        No push is possible — either home covers or away covers.

    For whole-number handicaps (e.g., ±1, ±2):
        Push (stake returned) when adjusted result is exactly zero.

    Args:
        lam_h:     Expected home goals.
        lam_a:     Expected away goals.
        handicap:  Applied to the home team's score (e.g., -1.5, +1, 0).

    Returns:
        {
            "handicap":       the input handicap value,
            "p_home_covers":  P(home wins after handicap adjustment),
            "p_push":         P(exact tie after adjustment — only for whole numbers),
            "p_away_covers":  P(away wins after handicap adjustment),
        }

    Examples:
        # Norway -1.5 (must win by 2+)
        >>> asian_handicap(1.845, 1.244, -1.5)
        {'handicap': -1.5, 'p_home_covers': 0.2641, 'p_push': 0.0, 'p_away_covers': 0.7359}

        # Senegal +1.5 (covers if Senegal lose by 1 or better)
        # This is the mirror of the above (p_away_covers of -1.5 = p_home_covers of +1.5)
        >>> asian_handicap(1.845, 1.244, +1.5)
        {'handicap': 1.5, 'p_home_covers': 0.7359, 'p_push': 0.0, 'p_away_covers': 0.2641}
    """
    matrix = build_dc_matrix(lam_h, lam_a)
    n      = len(matrix)

    p_home_covers = 0.0
    p_push        = 0.0
    p_away_covers = 0.0

    for h in range(n):
        for a in range(n):
            # Adjusted net: positive means home covered
            net = (h - a) + handicap
            if net > 0:
                p_home_covers += matrix[h][a]
            elif net == 0:
                p_push        += matrix[h][a]
            else:
                p_away_covers += matrix[h][a]

    return {
        "handicap":       handicap,
        "p_home_covers":  round(p_home_covers, 4),
        "p_push":         round(p_push, 4),
        "p_away_covers":  round(p_away_covers, 4),
    }


# ---------------------------------------------------------------------------
# Both Teams To Score (BTTS)
# ---------------------------------------------------------------------------

def btts_probability(lam_h: float, lam_a: float) -> dict:
    """
    P(both teams score at least one goal) and its complement.

    One of the most popular football betting markets.  Calculated from the
    DC matrix: P(BTTS) = 1 − P(home scores 0) − P(away scores 0) + P(0-0).
    (Inclusion-exclusion to avoid double-counting the 0-0 cell.)

    Args:
        lam_h:  Expected home goals.
        lam_a:  Expected away goals.

    Returns:
        {"p_yes": float, "p_no": float}
    """
    matrix = build_dc_matrix(lam_h, lam_a)
    n      = len(matrix)

    p_btts_yes = sum(
        matrix[h][a]
        for h in range(n)
        for a in range(n)
        if h >= 1 and a >= 1
    )
    return {
        "p_yes": round(p_btts_yes, 4),
        "p_no":  round(1.0 - p_btts_yes, 4),
    }


# ---------------------------------------------------------------------------
# Consolidated result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MarketResult:
    """
    All sub-market probabilities for a single match, computed together
    from one build_dc_matrix() call per function.

    Attributes
    ----------
    home_team, away_team : team names for display
    lam_h, lam_a        : λ values used (post-blend, post-DC)
    ou                  : dict keyed by line → {p_over, p_under, expected_goals}
    goal_diff           : dict keyed by integer GD → probability
    handicaps           : list of asian_handicap result dicts
    btts                : {p_yes, p_no}
    """
    home_team:  str
    away_team:  str
    lam_h:      float
    lam_a:      float
    sum_goals:  dict[str, float]          = field(default_factory=dict)
    ou:         dict[float, dict]         = field(default_factory=dict)
    goal_diff:  dict[int, float]          = field(default_factory=dict)
    handicaps:  list[dict]                = field(default_factory=list)
    btts:       dict                      = field(default_factory=dict)

    def summary(self) -> str:
        """
        Return a formatted multi-line string suitable for pipeline logs.
        """
        lines = [
            f"[markets] {self.home_team} vs {self.away_team}"
            f"  (λ: {self.lam_h:.2f} / {self.lam_a:.2f})",
        ]

        # Sum-goals distribution
        if self.sum_goals:
            sg = self.sum_goals
            lines.append(
                f"  Sum Goals:  0-1={sg.get('0-1',0):.1%}"
                f"  2-3={sg.get('2-3',0):.1%}"
                f"  4+={sg.get('+4',0):.1%}"
            )

        # BTTS
        lines.append(
            f"  BTTS:  yes={self.btts.get('p_yes',0):.1%}"
            f"  no={self.btts.get('p_no',0):.1%}"
        )

        # Goal difference (top 5 most likely)
        lines.append("  Goal Difference (top 5):")
        sorted_gd = sorted(self.goal_diff.items(), key=lambda kv: -kv[1])[:5]
        for gd, prob in sorted_gd:
            label = (
                f"{self.home_team} +{gd}" if gd > 0
                else ("Draw"              if gd == 0
                else f"{self.away_team} +{-gd}")
            )
            lines.append(f"    GD={gd:+d}  ({label:<30}) {prob:.1%}")

        # Asian handicaps
        if self.handicaps:
            lines.append("  Asian Handicap (home team):")
            for h in self.handicaps:
                push_str = f"  push={h['p_push']:.1%}" if h["p_push"] > 0 else ""
                lines.append(
                    f"    {h['handicap']:+.1f}  "
                    f"home={h['p_home_covers']:.1%}  "
                    f"away={h['p_away_covers']:.1%}"
                    f"{push_str}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def calculate_all_markets(
    lam_h: float,
    lam_a: float,
    home_team: str = "Home",
    away_team: str = "Away",
    ou_lines: Optional[list[float]] = None,
    handicap_lines: Optional[list[float]] = None,
) -> MarketResult:
    """
    Compute all sub-markets in one call and return a MarketResult.

    Args:
        lam_h:           Expected home goals (post-blend, post-DC).
        lam_a:           Expected away goals.
        home_team:       Display name for logs.
        away_team:       Display name for logs.
        ou_lines:        O/U lines to evaluate (default [1.5, 2.5, 3.5, 4.5]).
        handicap_lines:  Home team handicap lines (default [-1.5, -1.0, -0.5, 0, +0.5, +1.0, +1.5]).

    Returns:
        MarketResult with all markets populated.

    Example:
        markets = calculate_all_markets(1.845, 1.244, "Norway", "Senegal")
        print(markets.summary())
    """
    if ou_lines is None:
        ou_lines = [1.5, 2.5, 3.5, 4.5]
    if handicap_lines is None:
        handicap_lines = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

    return MarketResult(
        home_team  = home_team,
        away_team  = away_team,
        lam_h      = lam_h,
        lam_a      = lam_a,
        sum_goals  = sum_goals_distribution(lam_h, lam_a),
        ou         = ou_probabilities(lam_h, lam_a, ou_lines),
        goal_diff  = goal_diff_distribution(lam_h, lam_a),
        handicaps  = [asian_handicap(lam_h, lam_a, h) for h in handicap_lines],
        btts       = btts_probability(lam_h, lam_a),
    )
