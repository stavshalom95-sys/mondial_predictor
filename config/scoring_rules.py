"""
Single source of truth for tournament scoring rules and tiebreakers.
"""
from enum import Enum


class TournamentStage(Enum):
    GROUP_STAGE   = "שלב הבתים"
    ROUND_OF_32   = "32 האחרונות"
    ROUND_OF_16   = "שמינית גמר"
    QUARTER_FINAL = "רבע גמר"
    SEMI_FINAL    = "חצי גמר"
    THIRD_PLACE   = "מקום שלישי"
    FINAL         = "הגמר הגדול"


# Points per stage: correct direction (1X2) vs exact score
SCORING: dict[TournamentStage, dict[str, int]] = {
    TournamentStage.GROUP_STAGE:   {"direction": 1, "exact": 3},
    TournamentStage.ROUND_OF_32:   {"direction": 2, "exact": 5},
    TournamentStage.ROUND_OF_16:   {"direction": 2, "exact": 5},
    TournamentStage.QUARTER_FINAL: {"direction": 4, "exact": 8},
    TournamentStage.SEMI_FINAL:    {"direction": 5, "exact": 10},
    TournamentStage.THIRD_PLACE:   {"direction": 5, "exact": 10},
    TournamentStage.FINAL:         {"direction": 8, "exact": 15},
}

# Long-term bets points
LONG_TERM_BET_POINTS = {
    "champion":   12,
    "top_scorer": 12,  # tiebreak: anyone who picked a player from the winning team gets full points
}

# Tiebreak priority (index 0 = most important)
TIEBREAK_ORDER = [
    "exact_score_count",        # 1st — most critical for the model's risk calc
    "correct_direction_count",
    "top_scorer_goals",
    "champion_correct",
    "registration_time",        # last, uncontrollable
]


def stage_value_multiplier(stage: TournamentStage) -> float:
    """
    Ratio of exact-score points at this stage vs group stage.
    Used by strategy_advisor to scale the adaptive threshold.
    """
    base = SCORING[TournamentStage.GROUP_STAGE]["exact"]
    return SCORING[stage]["exact"] / base
