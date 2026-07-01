"""
Current tournament standings and long-term bets.

MY_CURRENT_STATE["matches_remaining"] is a default fallback only —
it is overridden at every run by data_pipeline.matches_remaining_in_tournament().
"""

MY_CURRENT_STATE: dict = {
    "my_points":       40,     # updated 2026-07-01: user is in 4th place, 5 pts behind leader
    "leader_points":   45,     # gap = 5 → trailing → aggressive exact-score mode
    "leader_name":     "leader",
    "matches_remaining": 10,   # runtime default; replaced by data_pipeline
}

LONG_TERM_BETS: dict = {
    "my_champion_pick":      "Spain",
    "leader_champion_pick":  "Spain",    # same -> no relative advantage/disadvantage here
    "my_top_scorer_pick":    "Harry Kane",
    "leader_top_scorer_pick": "Mbappe",  # asymmetric: my upside depends on Kane scoring
}
