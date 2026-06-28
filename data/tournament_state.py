"""
Current tournament standings and long-term bets.

MY_CURRENT_STATE["matches_remaining"] is a default fallback only —
it is overridden at every run by data_pipeline.matches_remaining_in_tournament().
"""

MY_CURRENT_STATE: dict = {
    "my_points":       40,     # updated 2026-06-28: user is in 1st place
    "leader_points":   36,     # user leads — gap < 0 → SAFE strategy (protect lead)
    "leader_name":     "me",
    "matches_remaining": 10,   # runtime default; replaced by data_pipeline
}

LONG_TERM_BETS: dict = {
    "my_champion_pick":      "Spain",
    "leader_champion_pick":  "Spain",    # same -> no relative advantage/disadvantage here
    "my_top_scorer_pick":    "Harry Kane",
    "leader_top_scorer_pick": "Mbappe",  # asymmetric: my upside depends on Kane scoring
}
