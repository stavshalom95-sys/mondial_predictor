"""
Poisson model: calibrate lambda_home / lambda_away from true 1X2 probabilities,
then build a full (home_goals x away_goals) probability matrix.

Calibration uses a two-pass grid search:
  1. Coarse pass (step 0.1) over a plausible range.
  2. Fine pass (step 0.02) in a ±0.10 window around the coarse best.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from core.odds_converter import TrueProbs1X2, TrueProbsOU

MAX_GOALS = 8


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _poisson_pmf(lam: float, k: int) -> float:
    """P(X=k) for X ~ Poisson(lam)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _build_matrix(lh: float, la: float) -> list[list[float]]:
    """
    Returns matrix[h][a] = P(home scores h, away scores a).
    Dimensions: (MAX_GOALS+1) x (MAX_GOALS+1).
    """
    return [
        [_poisson_pmf(lh, h) * _poisson_pmf(la, a) for a in range(MAX_GOALS + 1)]
        for h in range(MAX_GOALS + 1)
    ]


def _matrix_1x2(matrix: list[list[float]]) -> tuple[float, float, float]:
    """Aggregate matrix into (p_home_win, p_draw, p_away_win)."""
    p_home = p_draw = p_away = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = matrix[h][a]
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
    return p_home, p_draw, p_away


def _loss(ph: float, pd: float, pa: float, target: TrueProbs1X2) -> float:
    return (ph - target.home) ** 2 + (pd - target.draw) ** 2 + (pa - target.away) ** 2


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class ScoreProb:
    home_goals: int
    away_goals: int
    probability: float

    def __str__(self) -> str:
        return f"{self.home_goals}:{self.away_goals} ({self.probability*100:.1f}%)"


@dataclass
class PoissonMatchModel:
    lambda_home: float
    lambda_away: float
    _matrix: list[list[float]] = field(repr=False)

    def top_n(self, n: int) -> list[ScoreProb]:
        """Return the n most likely scorelines, sorted descending by probability."""
        scores = [
            ScoreProb(h, a, self._matrix[h][a])
            for h in range(MAX_GOALS + 1)
            for a in range(MAX_GOALS + 1)
        ]
        return sorted(scores, key=lambda s: s.probability, reverse=True)[:n]

    def probability_of(self, home_goals: int, away_goals: int) -> float:
        if 0 <= home_goals <= MAX_GOALS and 0 <= away_goals <= MAX_GOALS:
            return self._matrix[home_goals][away_goals]
        return 0.0

    def p_home_win(self) -> float:
        return sum(self._matrix[h][a] for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1) if h > a)

    def p_draw(self) -> float:
        return sum(self._matrix[h][a] for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1) if h == a)

    def p_away_win(self) -> float:
        return sum(self._matrix[h][a] for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1) if h < a)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(
    true_probs: TrueProbs1X2,
    ou_probs: Optional[TrueProbsOU] = None,
    avg_total_goals_hint: float = 2.6,
) -> PoissonMatchModel:
    """
    Find (lambda_home, lambda_away) that best reproduces true_probs.

    If ou_probs is provided, use it to estimate total expected goals:
        total_est = line + (p_over - p_under) * 1.2
    Otherwise fall back to avg_total_goals_hint (default 2.6).
    """
    if ou_probs is not None:
        total_est = ou_probs.line + (ou_probs.p_over - ou_probs.p_under) * 1.2
    else:
        total_est = avg_total_goals_hint

    total_est = max(total_est, 0.5)  # safety floor

    # --- Coarse grid search (step = 0.1) ---
    best_loss = float("inf")
    best_lh = best_la = 1.0

    coarse_step = 0.1
    # Search from 0.1 to min(6.0, 3*total_est)
    max_lam = min(6.0, total_est * 3)
    coarse_vals = [round(i * coarse_step, 4) for i in range(1, int(max_lam / coarse_step) + 1)]

    for lh in coarse_vals:
        for la in coarse_vals:
            # Prune: skip pairs far from expected total
            if abs(lh + la - total_est) > total_est * 1.0:
                continue
            matrix = _build_matrix(lh, la)
            ph, pd, pa = _matrix_1x2(matrix)
            loss = _loss(ph, pd, pa, true_probs)
            if loss < best_loss:
                best_loss = loss
                best_lh, best_la = lh, la

    # --- Fine grid search (step = 0.02) around coarse best ---
    fine_deltas = [round(i * 0.02, 4) for i in range(-5, 6)]  # -0.10 to +0.10
    for dlh in fine_deltas:
        for dla in fine_deltas:
            lh = round(best_lh + dlh, 4)
            la = round(best_la + dla, 4)
            if lh <= 0 or la <= 0:
                continue
            matrix = _build_matrix(lh, la)
            ph, pd, pa = _matrix_1x2(matrix)
            loss = _loss(ph, pd, pa, true_probs)
            if loss < best_loss:
                best_loss = loss
                best_lh, best_la = lh, la

    matrix = _build_matrix(best_lh, best_la)
    return PoissonMatchModel(lambda_home=best_lh, lambda_away=best_la, _matrix=matrix)
