"""
strategy_advisor.py — The competitive decision layer.

Key principle: in a friends' tournament you're competing against the distribution
of your opponents' guesses, not against statistical reality. If everyone picks
the consensus, nobody closes the gap when it lands.

Decision logic:
  - Compute gap_per_match = (leader_points - my_points) / matches_remaining
  - Compute adjusted_threshold = BASE_THRESHOLD / stage_value_multiplier(stage)
  - If gap_per_match <= adjusted_threshold  -> SAFE (consensus pick)
  - Else                                    -> CONTRARIAN (differentiated pick)

Tiebreak boost: exact-score picks serve double duty (raw points + tiebreak #1),
so their EV is multiplied by TIEBREAK_BOOST = 1.15.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config.scoring_rules import TournamentStage, SCORING, stage_value_multiplier
from core.poisson_engine import PoissonMatchModel, ScoreProb

TIEBREAK_BOOST = 1.15
POINT_GAP_THRESHOLD_BASE = 2.0  # gap-per-match above which we go contrarian (group-stage equivalent)
_CONSENSUS_TOP_N = 2             # top-N Poisson picks counted as "consensus"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Strategy(Enum):
    SAFE       = "שמרני"
    CONTRARIAN = "קונטרארי"


@dataclass
class TournamentContext:
    my_points:         int
    leader_points:     int
    matches_remaining: int          # overridden at runtime by data_pipeline
    current_stage:     TournamentStage = TournamentStage.GROUP_STAGE
    standings_source:  str             = "fallback"  # "live" | "fallback"
    leader_name:       str             = ""           # name of current leader
    my_rank:           int             = 0            # my rank in the group (0 = unknown)
    second_name:       str             = ""           # 2nd-place participant name
    second_points:     int             = 0            # 2nd-place score (used when I lead)

    @property
    def point_gap(self) -> int:
        return self.leader_points - self.my_points

    @property
    def gap_per_match(self) -> float:
        if self.matches_remaining <= 0:
            # No matches left → nothing to chase. Return 0 so the advisor
            # always picks SAFE (protect whatever lead or position we have).
            return 0.0
        return self.point_gap / self.matches_remaining


@dataclass
class StrategyRecommendation:
    strategy:                  Strategy
    recommended_pick:          ScoreProb
    reasoning:                 str
    alternative_safe_pick:     ScoreProb
    expected_value_safe:       float
    expected_value_contrarian: float
    stage:                     TournamentStage
    points_if_exact:           int
    points_if_direction_only:  int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_likely_consensus_pick(
    score: ScoreProb,
    model: PoissonMatchModel,
    top_n_for_consensus: int = _CONSENSUS_TOP_N,
) -> bool:
    """True if `score` falls in the top-N most-likely Poisson outcomes."""
    top = model.top_n(top_n_for_consensus)
    return any(s.home_goals == score.home_goals and s.away_goals == score.away_goals for s in top)


def find_contrarian_candidate(
    model: PoissonMatchModel,
    min_probability: float = 0.05,
    top_k_to_scan: int = 20,
) -> Optional[ScoreProb]:
    """
    Scan the top_k_to_scan most-likely scorelines.
    Return the first one that is:
      (a) NOT consensus (not in top-2 Poisson picks), and
      (b) above the min_probability floor (not a long-shot).

    Falls back by softening the probability floor: 0.05 -> 0.03 -> 0.015.
    Returns None only if truly no candidate found.
    """
    candidates = model.top_n(top_k_to_scan)
    for threshold in [min_probability, 0.03, 0.015]:
        for score in candidates:
            if not _is_likely_consensus_pick(score, model) and score.probability >= threshold:
                return score
    return None


def _p_direction(score: ScoreProb, model: PoissonMatchModel) -> float:
    """P(match outcome direction matches score's direction)."""
    if score.home_goals > score.away_goals:
        return model.p_home_win()
    elif score.home_goals == score.away_goals:
        return model.p_draw()
    else:
        return model.p_away_win()


def _expected_value(
    score: ScoreProb,
    model: PoissonMatchModel,
    stage: TournamentStage,
) -> float:
    """
    EV = P(exact) * exact_pts * TIEBREAK_BOOST
       + (P(direction) - P(exact)) * direction_pts
    """
    exact_pts    = SCORING[stage]["exact"]
    direction_pts = SCORING[stage]["direction"]

    p_exact = score.probability
    p_dir   = _p_direction(score, model)

    return p_exact * exact_pts * TIEBREAK_BOOST + (p_dir - p_exact) * direction_pts


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def recommend(
    model:   PoissonMatchModel,
    context: TournamentContext,
    stage:   TournamentStage,
) -> StrategyRecommendation:
    """
    Return a StrategyRecommendation for one match given the current standings.

    Adaptive threshold table (point_gap_threshold_base = 2.0, gap_per_match = 1.5):
      Group stage   (1.0x) -> adjusted 2.00 -> 1.5 <= 2.00 -> SAFE
      Round of 16   (1.67x) -> adjusted 1.20 -> 1.5 > 1.20  -> CONTRARIAN
      Quarter final (2.67x) -> adjusted 0.75 -> 1.5 > 0.75  -> CONTRARIAN
      Semi final    (3.33x) -> adjusted 0.60 -> 1.5 > 0.60  -> CONTRARIAN
      Final         (5.00x) -> adjusted 0.40 -> 1.5 > 0.40  -> CONTRARIAN
    """
    multiplier         = stage_value_multiplier(stage)
    adjusted_threshold = POINT_GAP_THRESHOLD_BASE / multiplier

    safe_pick        = model.top_n(1)[0]
    contrarian_pick  = find_contrarian_candidate(model)
    if contrarian_pick is None:
        contrarian_pick = model.top_n(3)[-1]  # fallback: 3rd most likely

    ev_safe        = _expected_value(safe_pick, model, stage)
    ev_contrarian  = _expected_value(contrarian_pick, model, stage)

    # Decision — Contrarian logic suspended (Measurement-First protocol June 2026).
    # Always pick the modal (consensus) score until we have enough calibration data
    # to validate whether contrarian picks improve expected score in this 10-person pool.
    chosen_strategy = Strategy.SAFE
    recommended     = safe_pick

    exact_pts     = SCORING[stage]["exact"]
    direction_pts = SCORING[stage]["direction"]

    reasoning_lines = [
        f"פער: {context.point_gap} נק' | משחקים נותרים: {context.matches_remaining}",
        f"פער-למשחק: {context.gap_per_match:.2f} | סף מתואם: {adjusted_threshold:.2f} (מכפיל שלב: {multiplier:.2f}x)",
        f"אסטרטגיה נבחרת: {chosen_strategy.value}",
        f"המלצה: {recommended.home_goals}:{recommended.away_goals} ({recommended.probability*100:.1f}% סיכוי)",
        f"EV שמרני: {ev_safe:.3f} | EV קונטרארי: {ev_contrarian:.3f}",
    ]
    if chosen_strategy == Strategy.CONTRARIAN:
        reasoning_lines.append(
            f"קונצנזוס (הימנענו ממנו): {safe_pick.home_goals}:{safe_pick.away_goals} "
            f"({safe_pick.probability*100:.1f}%)"
        )

    return StrategyRecommendation(
        strategy=chosen_strategy,
        recommended_pick=recommended,
        reasoning="\n".join(reasoning_lines),
        alternative_safe_pick=safe_pick,
        expected_value_safe=ev_safe,
        expected_value_contrarian=ev_contrarian,
        stage=stage,
        points_if_exact=exact_pts,
        points_if_direction_only=direction_pts,
    )
