"""
Poisson model: calibrate lambda_home / lambda_away from true 1X2 probabilities,
then build a full (home_goals x away_goals) probability matrix.

Calibration uses a two-pass grid search:
  1. Coarse pass (step 0.1) over a plausible range.
  2. Fine pass (step 0.02) in a ±0.10 window around the coarse best.

Enhancements (v2):
  - Dixon-Coles correction (_dc_tau, build_dc_matrix, calibrate_dc):
    Adjusts the four low-scoring cells (0-0, 1-0, 0-1, 1-1) to correct
    the independence assumption. Standard Poisson over-estimates clean
    sheets and under-estimates 1-1 draws; DC fixes this.

  - Tournament scaling (tournament_scale):
    Scales λ values when they were estimated from raw observed goals rather
    than market odds. Do NOT call when calibrate() was used — market odds
    already embed the current tournament scoring pace.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from core.odds_converter import TrueProbs1X2, TrueProbsOU

MAX_GOALS = 8

# Long-run FIFA World Cup average goals/game across all editions.
# Used as the denominator for tournament scaling.
HISTORICAL_WC_AVG_GOALS: float = 2.64

# Dixon-Coles ρ (rho): negative correlation between home/away goal counts.
# Literature consensus: -0.10 to -0.15.  -0.13 is the accepted midpoint.
_DC_RHO: float = -0.13


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
    Pure independent Poisson — no Dixon-Coles correction.
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
# Dixon-Coles correction
# ---------------------------------------------------------------------------

def _dc_tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    """
    Dixon-Coles τ (tau) correction factor for score (h, a).

    Only four low-scoring cells are adjusted; all others return 1.0.
    With rho = -0.13 and typical WC λ values (lh≈1.9, la≈1.1):

        0-0: τ = 1 - lh*la*rho ≈ 1.277  (boosted — more common than Poisson predicts)
        1-0: τ = 1 + la*rho    ≈ 0.854  (suppressed — less common)
        0-1: τ = 1 + lh*rho    ≈ 0.753  (suppressed)
        1-1: τ = 1 - rho       ≈ 1.130  (boosted)
    """
    if   h == 0 and a == 0:  return 1.0 - lh * la * rho
    elif h == 1 and a == 0:  return 1.0 + la * rho
    elif h == 0 and a == 1:  return 1.0 + lh * rho
    elif h == 1 and a == 1:  return 1.0 - rho
    else:                    return 1.0


def build_dc_matrix(
    lh: float,
    la: float,
    rho: float = _DC_RHO,
) -> list[list[float]]:
    """
    Build a Dixon-Coles corrected score probability matrix.

    Applies _dc_tau() to the four low-score cells then normalises so
    all cells still sum to 1.0.

    Args:
        lh:  Expected home goals (λ_home).
        la:  Expected away goals (λ_away).
        rho: DC correlation parameter (default -0.13).

    Returns:
        Normalised (MAX_GOALS+1) × (MAX_GOALS+1) probability matrix.
    """
    raw = [
        [
            _poisson_pmf(lh, h) * _poisson_pmf(la, a) * _dc_tau(h, a, lh, la, rho)
            for a in range(MAX_GOALS + 1)
        ]
        for h in range(MAX_GOALS + 1)
    ]
    total = sum(raw[h][a] for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1))
    if total <= 0:
        return raw   # safety fallback for degenerate inputs
    return [[raw[h][a] / total for a in range(MAX_GOALS + 1)] for h in range(MAX_GOALS + 1)]


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
        return sum(
            self._matrix[h][a]
            for h in range(MAX_GOALS + 1)
            for a in range(MAX_GOALS + 1)
            if h > a
        )

    def p_draw(self) -> float:
        return sum(
            self._matrix[h][a]
            for h in range(MAX_GOALS + 1)
            for a in range(MAX_GOALS + 1)
            if h == a
        )

    def p_away_win(self) -> float:
        return sum(
            self._matrix[h][a]
            for h in range(MAX_GOALS + 1)
            for a in range(MAX_GOALS + 1)
            if h < a
        )


# ---------------------------------------------------------------------------
# Calibration (unchanged — market-odds path)
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

    Returns a PoissonMatchModel with a raw (non-DC) matrix.
    Use calibrate_dc() for the Dixon-Coles corrected version.
    """
    if ou_probs is not None:
        total_est = ou_probs.line + (ou_probs.p_over - ou_probs.p_under) * 1.2
    else:
        total_est = avg_total_goals_hint

    total_est = max(total_est, 0.5)

    best_loss = float("inf")
    best_lh = best_la = 1.0

    coarse_step = 0.1
    max_lam     = min(6.0, total_est * 3)
    coarse_vals = [round(i * coarse_step, 4) for i in range(1, int(max_lam / coarse_step) + 1)]

    for lh in coarse_vals:
        for la in coarse_vals:
            if abs(lh + la - total_est) > total_est * 1.0:
                continue
            matrix = _build_matrix(lh, la)
            ph, pd, pa = _matrix_1x2(matrix)
            loss = _loss(ph, pd, pa, true_probs)
            if loss < best_loss:
                best_loss = loss
                best_lh, best_la = lh, la

    fine_deltas = [round(i * 0.02, 4) for i in range(-5, 6)]
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


# ---------------------------------------------------------------------------
# Enhanced calibration — Dixon-Coles corrected (v2)
# ---------------------------------------------------------------------------

def calibrate_dc(
    true_probs: TrueProbs1X2,
    ou_probs: Optional[TrueProbsOU] = None,
    avg_total_goals_hint: float = 2.6,
    rho: float = _DC_RHO,
) -> PoissonMatchModel:
    """
    Drop-in replacement for calibrate() that applies Dixon-Coles correction.

    Runs the identical two-pass grid search to find optimal λ values, then
    replaces the raw Poisson matrix with a DC-corrected, normalised matrix.

    In main.py, swap:
        model = calibrate(true_probs, ou_probs)
    for:
        model = calibrate_dc(true_probs, ou_probs)

    Args:
        true_probs: TrueProbs1X2 from odds_converter (overround removed).
        ou_probs:   Optional over/under odds for total-goals estimation.
        avg_total_goals_hint: Fallback total if ou_probs not provided.
        rho: Dixon-Coles correlation parameter (default -0.13).

    Returns:
        PoissonMatchModel with DC-corrected probability matrix.
    """
    base   = calibrate(true_probs, ou_probs, avg_total_goals_hint)
    matrix = build_dc_matrix(base.lambda_home, base.lambda_away, rho)
    return PoissonMatchModel(
        lambda_home=base.lambda_home,
        lambda_away=base.lambda_away,
        _matrix=matrix,
    )


# ---------------------------------------------------------------------------
# Tournament scaling — raw-observation λ only (v2)
# ---------------------------------------------------------------------------

def tournament_scale(
    model: PoissonMatchModel,
    observed_tournament_avg: float,
    from_market_odds: bool = True,
    use_dc: bool = True,
    rho: float = _DC_RHO,
) -> PoissonMatchModel:
    """
    Scale λ values to match the current tournament's observed scoring pace.

    ⚠️  Only call when λ was derived from raw observed goals (e.g. manual
        estimates from standings data).  When calibrate() / calibrate_dc()
        is used, market odds already reflect the scoring environment —
        applying this on top would double-count.

    Scaling formula:
        scale = observed_tournament_avg / HISTORICAL_WC_AVG_GOALS
        λ_scaled = λ_raw × scale

    Example (WC 2026 as of 2026-06-21):
        observed_tournament_avg = 3.02
        scale  = 3.02 / 2.64 = 1.144
        Norway 1.80 → 2.06,  Senegal 0.90 → 1.03

    Args:
        model:                   PoissonMatchModel to scale.
        observed_tournament_avg: Current tournament goals/game.
        from_market_odds:        If True, return model unchanged (no-op).
        use_dc:                  If True, rebuild with DC matrix (recommended).
        rho:                     DC rho (only used when use_dc=True).

    Returns:
        New PoissonMatchModel with scaled λ and updated matrix.
    """
    if from_market_odds:
        return model   # no-op: market odds self-correct for scoring pace

    scale  = observed_tournament_avg / HISTORICAL_WC_AVG_GOALS
    lh     = round(model.lambda_home * scale, 3)
    la     = round(model.lambda_away * scale, 3)
    matrix = build_dc_matrix(lh, la, rho) if use_dc else _build_matrix(lh, la)
    return PoissonMatchModel(lambda_home=lh, lambda_away=la, _matrix=matrix)
