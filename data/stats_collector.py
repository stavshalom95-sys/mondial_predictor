"""
data/stats_collector.py — Team rolling-form collector.

Computes per-team last-N match averages from the WC schedule JSON that is
already fetched by fetch_schedule.py (zero extra API calls needed).

For each team the module returns:
  goals_scored_avg   — average goals scored per game (last N WC matches)
  goals_conceded_avg — average goals conceded per game (last N WC matches)
  corner_avg         — always WC_CORNER_AVG (football-data.org free tier
                       does not include corner counts; league average used)
  n_games            — number of finished WC matches found (may be < last_n
                       early in the tournament)

Public API:
    collect_team_form(raw_games, team_name, last_n=5) -> TeamForm
    build_form_cache(raw_games, last_n=5)             -> FormCache

Integration with main.py (after strength-model blending):

    _form_cache = build_form_cache(raw_games_from_api)

    # In the match loop:
    _h_form = _form_cache.get(match.home_team)
    _a_form = _form_cache.get(match.away_team)
    if _h_form and _h_form.n_games > 0:
        _scale = _h_form.goals_scored_avg / _form_cache.tournament_avg
        lam_h  = round((1 - FORM_BLEND_WEIGHT) * lam_h + FORM_BLEND_WEIGHT * lam_h * _scale, 3)
    # (same for lam_a / _a_form)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

FORM_GAMES:       int   = 5     # last N finished WC matches per team
FORM_BLEND_WEIGHT: float = 0.15  # 15% form  +  85% market/strength — conservative blend

# football-data.org free tier has no corner data → always use this tournament average.
# WC 2026 baseline: ~9.5 total corners per match = 4.75 per team per game.
WC_CORNER_AVG: float = 4.75


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Strip leading emoji-flag token and lowercase — mirrors motivation._norm."""
    name = name.strip()
    if name and not name[0].isascii():
        parts = name.split(None, 1)
        name = parts[1] if len(parts) > 1 else ""
    return name.lower().strip()


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TeamForm:
    team_name:          str
    n_games:            int     # WC finished matches found (may be < FORM_GAMES early on)
    goals_scored_avg:   float   # goals this team scored / game (last N WC games)
    goals_conceded_avg: float   # goals this team conceded / game (last N WC games)
    corner_avg:         float   # always WC_CORNER_AVG — no corner data from free API

    def __str__(self) -> str:
        if self.n_games == 0:
            return f"{self.team_name}: no WC data yet"
        return (
            f"{self.team_name}: scored={self.goals_scored_avg:.2f}/g  "
            f"conceded={self.goals_conceded_avg:.2f}/g  "
            f"(last {self.n_games} WC games)"
        )


@dataclass
class FormCache:
    """Per-team form data plus the current tournament goal average."""
    _cache:         dict[str, TeamForm]
    tournament_avg: float   # total goals / (2 × finished matches) for this WC so far

    def get(self, team_name: str) -> Optional[TeamForm]:
        """Case-insensitive lookup; returns None if team not in cache."""
        return self._cache.get(_norm(team_name))

    def __len__(self) -> int:
        return len(self._cache)


# ── Core logic ────────────────────────────────────────────────────────────────

def collect_team_form(
    raw_games: list[dict],
    team_name: str,
    last_n:    int = FORM_GAMES,
) -> TeamForm:
    """
    Compute rolling-form stats for one team from the WC schedule JSON.

    Scans all finished ('final') matches, collects the team's results in
    chronological API order (football-data.org sorts ascending by date),
    keeps the most recent `last_n`, and returns per-game averages.

    Args:
        raw_games:  List of raw match dicts from fetch_schedule.py output.
        team_name:  Full team name as it appears in the schedule (e.g. "Colombia").
        last_n:     Rolling window size (default 5).

    Returns:
        TeamForm with averaged stats and n_games=0 if team has no WC data.
    """
    team_norm = _norm(team_name)
    results: list[tuple[int, int]] = []   # (goals_scored, goals_conceded)

    for g in raw_games:
        if g.get("status") != "final":
            continue

        teams    = g.get("teams", {})
        home_key = g.get("home", "")
        away_key = g.get("away", "")

        home_name = teams.get(home_key, {}).get("name", home_key) if home_key else ""
        away_name = teams.get(away_key, {}).get("name", away_key) if away_key else ""

        is_home = _norm(home_name) == team_norm
        is_away = _norm(away_name) == team_norm
        if not is_home and not is_away:
            continue

        score = g.get("score", {})
        try:
            hg = int(score.get(home_key))
            ag = int(score.get(away_key))
        except (TypeError, ValueError):
            continue

        results.append((hg, ag) if is_home else (ag, hg))

    recent = results[-last_n:]   # last N in chronological order
    n      = len(recent)

    if n == 0:
        return TeamForm(
            team_name          = team_name,
            n_games            = 0,
            goals_scored_avg   = 0.0,
            goals_conceded_avg = 0.0,
            corner_avg         = WC_CORNER_AVG,
        )

    scored   = sum(s for s, _ in recent)
    conceded = sum(c for _, c in recent)
    return TeamForm(
        team_name          = team_name,
        n_games            = n,
        goals_scored_avg   = round(scored   / n, 3),
        goals_conceded_avg = round(conceded / n, 3),
        corner_avg         = WC_CORNER_AVG,
    )


def build_form_cache(
    raw_games: list[dict],
    last_n:    int = FORM_GAMES,
) -> FormCache:
    """
    Build form stats for every team in the schedule (no extra API calls).

    Also computes the current WC tournament average (goals per team per game)
    from all finished matches, used as the normalization denominator when
    computing form_scale in main.py:

        form_scale = team.goals_scored_avg / form_cache.tournament_avg

    Runtime: O(teams × games) — fast (<10 ms for a 104-game WC schedule).

    Returns:
        FormCache ready for per-match lookups.
    """
    # ── Collect unique team names ─────────────────────────────────────────────
    team_names: dict[str, str] = {}   # norm_key → display_name
    for g in raw_games:
        teams = g.get("teams", {})
        for key in (g.get("home", ""), g.get("away", "")):
            if not key:
                continue
            name = teams.get(key, {}).get("name", "")
            if name:
                team_names[_norm(name)] = name

    # ── Tournament average goals/team/game ────────────────────────────────────
    total_goals = 0
    total_games = 0
    for g in raw_games:
        if g.get("status") != "final":
            continue
        score = g.get("score", {})
        hk    = g.get("home", "")
        ak    = g.get("away", "")
        try:
            total_goals += int(score.get(hk)) + int(score.get(ak))
            total_games += 1
        except (TypeError, ValueError):
            pass

    t_avg = round(total_goals / (2 * total_games), 3) if total_games else 1.32

    # ── Build per-team form ───────────────────────────────────────────────────
    cache: dict[str, TeamForm] = {}
    for norm_key, display_name in team_names.items():
        form = collect_team_form(raw_games, display_name, last_n)
        cache[norm_key] = form

    print(
        f"[form] Built form cache: {len(cache)} teams  |  "
        f"tournament avg = {t_avg:.2f} g/team/game  |  "
        f"last {last_n} WC games per team"
    )
    return FormCache(_cache=cache, tournament_avg=t_avg)
