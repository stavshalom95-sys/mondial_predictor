"""
simulator.py — Monte Carlo + analytical Poisson score matrix.

simulate() returns a SimResult with two parallel columns of evidence:

  MC (Monte Carlo)          — 10,000 random Poisson draws, vectorized via numpy
  Poisson (analytical)      — exact Poisson PMF, reuses _build_matrix() from poisson_engine

At n_sims → ∞ the MC values converge to the analytical ones.  At 10k sims the
standard error is ~0.5 pp, so differences > 1.5 pp flag sampling noise.

ScoreGrid wraps the 5 × 5 (goals 0–4 each team) sub-matrix with helper methods
for display and the most-likely scorelines.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.poisson_engine import _build_matrix, MAX_GOALS

_RNG      = np.random.default_rng()   # module-level, thread-safe seeded generator
_MAX_DISP = 4                          # score grid shows 0–4 goals per team (5 × 5)


# ---------------------------------------------------------------------------
# Score Grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreGrid:
    """
    Analytical Poisson probability matrix (goals 0–4 each team).

    probs[h][a] = P(home scores h AND away scores a)
    Marginal probabilities outside the 0–4 window are captured in
    p_home_win / p_draw / p_away_win which use the full MAX_GOALS matrix.
    """
    probs:      tuple          # tuple[tuple[float, ...], ...], shape 5 × 5
    p_home_win: float
    p_draw:     float
    p_away_win: float
    lambda_home: float
    lambda_away: float

    # ------------------------------------------------------------------
    def top_scores(self, n: int = 5) -> list[tuple[int, int, float]]:
        """Return [(home_goals, away_goals, probability), ...] sorted descending."""
        cells = [
            (h, a, self.probs[h][a])
            for h in range(_MAX_DISP + 1)
            for a in range(_MAX_DISP + 1)
        ]
        return sorted(cells, key=lambda x: x[2], reverse=True)[:n]

    def most_likely_score(self) -> tuple[int, int]:
        h, a, _ = self.top_scores(1)[0]
        return h, a

    def as_dict_pct(self) -> list[dict]:
        """
        Serialize to a JSON-friendly list of {h, a, pct} for morning_picks.json.
        pct is rounded to 2 decimal places (percentage points).
        """
        return [
            {"h": h, "a": a, "pct": round(self.probs[h][a] * 100, 2)}
            for h, a, _ in self.top_scores(5)
        ]


# ---------------------------------------------------------------------------
# SimResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimResult:
    # ── Monte Carlo (random Poisson draws) ──────────────────────────────────
    p_home:  float     # fraction of sims where home won
    p_draw:  float
    p_away:  float
    n_sims:  int

    # ── Analytical Poisson (exact given λ) ──────────────────────────────────
    poisson_p_home:  float
    poisson_p_draw:  float
    poisson_p_away:  float

    # ── Score probability matrix (5 × 5) ────────────────────────────────────
    score_grid:  ScoreGrid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(home_xg: float, away_xg: float, n_sims: int = 10_000) -> SimResult:
    """
    Run MC simulation AND compute analytical Poisson score matrix for the same λ.

    home_xg / away_xg correspond to PoissonMatchModel.lambda_home / lambda_away.

    MC and Poisson results use *identical* λ values — differences reflect
    sampling variance only.  The Poisson values are the theoretically exact
    reference; the MC adds the score distribution via ScoreGrid.
    """
    # ── Monte Carlo (vectorized) ─────────────────────────────────────────────
    home_g = _RNG.poisson(home_xg, n_sims)
    away_g = _RNG.poisson(away_xg, n_sims)
    mc_home = float(np.mean(home_g > away_g))
    mc_draw = float(np.mean(home_g == away_g))
    mc_away = float(np.mean(home_g < away_g))

    # ── Analytical Poisson — reuse _build_matrix from poisson_engine ─────────
    full = _build_matrix(home_xg, away_xg)   # (MAX_GOALS+1) × (MAX_GOALS+1)

    p_h = sum(
        full[h][a]
        for h in range(MAX_GOALS + 1)
        for a in range(MAX_GOALS + 1)
        if h > a
    )
    p_d = sum(
        full[h][a]
        for h in range(MAX_GOALS + 1)
        for a in range(MAX_GOALS + 1)
        if h == a
    )
    p_a = 1.0 - p_h - p_d  # avoids rounding issues

    # ── Score grid (0–4 each team) ───────────────────────────────────────────
    grid = tuple(
        tuple(full[h][a] for a in range(_MAX_DISP + 1))
        for h in range(_MAX_DISP + 1)
    )
    score_grid = ScoreGrid(
        probs       = grid,
        p_home_win  = round(p_h, 4),
        p_draw      = round(p_d, 4),
        p_away_win  = round(p_a, 4),
        lambda_home = home_xg,
        lambda_away = away_xg,
    )

    return SimResult(
        p_home          = mc_home,
        p_draw          = mc_draw,
        p_away          = mc_away,
        n_sims          = n_sims,
        poisson_p_home  = round(p_h, 4),
        poisson_p_draw  = round(p_d, 4),
        poisson_p_away  = round(p_a, 4),
        score_grid      = score_grid,
    )
