"""
Data pipeline: parse the World Cup schedule from raw API dicts,
compute tournament state (matches remaining, upcoming games, etc.).

Expected raw_games format (matches both fetch_sports_data internal tool
and scripts/fetch_schedule.py from football-data.org):

{
  "id": "sr:sport_event:66456998",
  "status": "scheduled",          # "scheduled" | "live" | "final"
  "start_time": "2026-06-21T16:00:00+00:00",
  "home": "ESP",
  "away": "KSA",
  "teams": {
    "ESP": {"name": "Spain",        "abbreviation": "ESP"},
    "KSA": {"name": "Saudi Arabia", "abbreviation": "KSA"}
  },
  "score": {"ESP": 0, "KSA": 0}
}
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ScheduledMatch:
    match_id:       str
    home_team:      str
    away_team:      str
    start_time_utc: datetime
    status:         str            # "scheduled" | "live" | "final"
    home_score:     Optional[int]
    away_score:     Optional[int]

    def __str__(self) -> str:
        score = ""
        if self.home_score is not None and self.away_score is not None:
            score = f" {self.home_score}:{self.away_score}"
        return f"{self.home_team} vs {self.away_team}{score} [{self.status}]"


def parse_world_cup_schedule(raw_games: list[dict]) -> list[ScheduledMatch]:
    """Parse a list of raw API game dicts into ScheduledMatch objects."""
    matches: list[ScheduledMatch] = []
    for g in raw_games:
        teams    = g.get("teams", {})
        home_key = g.get("home", "")
        away_key = g.get("away", "")

        home_name = teams.get(home_key, {}).get("name", home_key)
        away_name = teams.get(away_key, {}).get("name", away_key)

        start_raw = g.get("start_time", "")
        try:
            start_time = datetime.fromisoformat(start_raw)
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            start_time = datetime.now(timezone.utc)

        score      = g.get("score", {})
        home_score = score.get(home_key)
        away_score = score.get(away_key)

        # Normalise None scores for non-final matches
        if home_score is not None:
            home_score = int(home_score)
        if away_score is not None:
            away_score = int(away_score)

        matches.append(ScheduledMatch(
            match_id       = str(g.get("id", "")),
            home_team      = home_name,
            away_team      = away_name,
            start_time_utc = start_time,
            status         = g.get("status", "scheduled"),
            home_score     = home_score,
            away_score     = away_score,
        ))

    return matches


def matches_remaining_in_tournament(
    all_matches: list[ScheduledMatch],
    as_of: Optional[datetime] = None,
) -> int:
    """
    Count matches that have not yet finished.
    A match is 'remaining' if its status is not 'final' AND its start time is in the future.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    return sum(
        1 for m in all_matches
        if m.status != "final" and m.start_time_utc > as_of
    )


def get_next_unplayed_matches(
    all_matches: list[ScheduledMatch],
    limit: int = 5,
) -> list[ScheduledMatch]:
    """Return the next `limit` scheduled (not yet started) matches, sorted by start time."""
    now = datetime.now(timezone.utc)
    upcoming = [m for m in all_matches if m.status == "scheduled" and m.start_time_utc > now]
    upcoming.sort(key=lambda m: m.start_time_utc)
    return upcoming[:limit]


def get_match_by_teams(
    all_matches: list[ScheduledMatch],
    home_team: str,
    away_team: str,
) -> Optional[ScheduledMatch]:
    """Case-insensitive lookup by team name."""
    h = home_team.lower()
    a = away_team.lower()
    for m in all_matches:
        if m.home_team.lower() == h and m.away_team.lower() == a:
            return m
    return None
