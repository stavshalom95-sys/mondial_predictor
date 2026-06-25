"""
core/tiebreaker.py — FIFA 2026 group-stage tiebreaker engine.

FIFA 2026 WC qualification (Group Stage):
  - 12 groups of 4 teams; 3 matchdays.
  - Top 2 teams from each group qualify directly (24 teams).
  - 8 best third-place teams from all 12 groups also qualify.
  - Total: 24 + 8 = 32 teams advance to the Round of 32.

FIFA Tiebreaker Order (strict priority when teams are level on points):
  1. Head-to-head points
  2. Head-to-head goal difference
  3. Head-to-head goals scored
  4. Overall group goal difference
  5. Overall group goals scored
  6. Fair play (disciplinary records) — treated as neutral here (not tracked)
  7. Drawing of lots — not resolvable in simulation

The same 5-step sequence is used to rank all 12 third-place teams for the
"8 best third-place" qualification path.

Public API:
    extract_h2h(team_a, team_b, completed_matches) -> (H2HRecord, H2HRecord)
    resolve_tiebreaker(team_a, team_b, row_a, row_b, completed_matches) -> TiebreakerResult
    build_tiebreaker_context(team_a, team_b, row_a, row_b, completed_matches, today_is_h2h) -> str
    rank_third_place_teams(all_groups) -> dict[str, int]
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Name normalisation (mirrors data/motivation._norm)
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Strip emoji flags, lowercase, remove diacritics."""
    name = name.strip()
    if name and not name[0].isascii():
        parts = name.split(None, 1)
        name = parts[1] if len(parts) > 1 else ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()


def _teams_match(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if na == nb or na in nb or nb in na:
        return True
    words_a = {w for w in na.split() if len(w) > 3}
    words_b = {w for w in nb.split() if len(w) > 3}
    return bool(words_a & words_b)


# ---------------------------------------------------------------------------
# Head-to-Head record
# ---------------------------------------------------------------------------

@dataclass
class H2HRecord:
    """Accumulated head-to-head stats for one team vs. a specific opponent."""
    team:   str
    pts:    int = 0    # points earned in H2H match(es)
    gf:     int = 0    # goals scored in H2H
    ga:     int = 0    # goals conceded in H2H
    played: int = 0    # H2H matches completed

    @property
    def gd(self) -> int:
        return self.gf - self.ga


def _apply_h2h(rec: H2HRecord, gf: int, ga: int) -> None:
    """Update a H2HRecord with one match result (from this team's perspective)."""
    rec.played += 1
    rec.gf     += gf
    rec.ga     += ga
    rec.pts    += 3 if gf > ga else (1 if gf == ga else 0)


def extract_h2h(
    team_a: str,
    team_b: str,
    completed_matches: list[dict],
) -> tuple[H2HRecord, H2HRecord]:
    """
    Return (H2HRecord for team_a, H2HRecord for team_b) from a list of
    completed match dicts.

    Accepts both schedule format (home_goals/away_goals) and history format
    (actual_home/actual_away).
    """
    rec_a = H2HRecord(team=team_a)
    rec_b = H2HRecord(team=team_b)

    for m in completed_matches:
        ht = m.get("home_team", "")
        at = m.get("away_team", "")
        hg = m.get("home_goals", m.get("actual_home"))
        ag = m.get("away_goals", m.get("actual_away"))
        if hg is None or ag is None:
            continue
        try:
            hg, ag = int(hg), int(ag)
        except (TypeError, ValueError):
            continue

        if _teams_match(team_a, ht) and _teams_match(team_b, at):
            _apply_h2h(rec_a, hg, ag)
            _apply_h2h(rec_b, ag, hg)
        elif _teams_match(team_a, at) and _teams_match(team_b, ht):
            _apply_h2h(rec_a, ag, hg)
            _apply_h2h(rec_b, hg, ag)

    return rec_a, rec_b


# ---------------------------------------------------------------------------
# Tiebreaker resolution
# ---------------------------------------------------------------------------

@dataclass
class TiebreakerResult:
    """Outcome of a FIFA tiebreaker resolution between two equal-points teams."""
    winner:     str
    loser:      str
    step:       int   # which FIFA step was decisive (1–7)
    step_name:  str   # human-readable label for the decisive step
    margin:     int   # advantage at the decisive step (pts or goals)
    h2h_played: int   # number of H2H matches already completed


_STEP_NAMES = {
    1: "H2H points",
    2: "H2H goal difference",
    3: "H2H goals scored",
    4: "Overall goal difference",
    5: "Overall goals scored",
    6: "Fair play (cards)",
    7: "Drawing of lots",
}


def resolve_tiebreaker(
    team_a: str,
    team_b: str,
    row_a: dict,
    row_b: dict,
    completed_matches: list[dict],
) -> TiebreakerResult:
    """
    Resolve the current FIFA 2026 tiebreaker between two equal-points teams.

    Parameters
    ----------
    team_a, team_b       : team names (raw, emoji-safe)
    row_a, row_b         : group table rows (goal_difference, goals_for, etc.)
    completed_matches    : list of finished match dicts (for H2H extraction)

    Returns TiebreakerResult with the current leader and the decisive step.
    H2H steps (1–3) are only evaluated when at least one H2H match was played.
    """
    h2h_a, h2h_b = extract_h2h(team_a, team_b, completed_matches)

    if h2h_a.played > 0:
        # Step 1: H2H points
        if h2h_a.pts != h2h_b.pts:
            w, l = (team_a, team_b) if h2h_a.pts > h2h_b.pts else (team_b, team_a)
            return TiebreakerResult(winner=w, loser=l, step=1,
                                    step_name=_STEP_NAMES[1],
                                    margin=abs(h2h_a.pts - h2h_b.pts),
                                    h2h_played=h2h_a.played)

        # Step 2: H2H goal difference
        if h2h_a.gd != h2h_b.gd:
            w, l = (team_a, team_b) if h2h_a.gd > h2h_b.gd else (team_b, team_a)
            return TiebreakerResult(winner=w, loser=l, step=2,
                                    step_name=_STEP_NAMES[2],
                                    margin=abs(h2h_a.gd - h2h_b.gd),
                                    h2h_played=h2h_a.played)

        # Step 3: H2H goals scored
        if h2h_a.gf != h2h_b.gf:
            w, l = (team_a, team_b) if h2h_a.gf > h2h_b.gf else (team_b, team_a)
            return TiebreakerResult(winner=w, loser=l, step=3,
                                    step_name=_STEP_NAMES[3],
                                    margin=abs(h2h_a.gf - h2h_b.gf),
                                    h2h_played=h2h_a.played)

    # Steps 4–5: Overall group stats (available regardless of H2H)
    gd_a = row_a.get("goal_difference", 0)
    gd_b = row_b.get("goal_difference", 0)
    if gd_a != gd_b:
        w, l = (team_a, team_b) if gd_a > gd_b else (team_b, team_a)
        return TiebreakerResult(winner=w, loser=l, step=4,
                                step_name=_STEP_NAMES[4],
                                margin=abs(gd_a - gd_b),
                                h2h_played=h2h_a.played)

    gf_a = row_a.get("goals_for", 0)
    gf_b = row_b.get("goals_for", 0)
    if gf_a != gf_b:
        w, l = (team_a, team_b) if gf_a > gf_b else (team_b, team_a)
        return TiebreakerResult(winner=w, loser=l, step=5,
                                step_name=_STEP_NAMES[5],
                                margin=abs(gf_a - gf_b),
                                h2h_played=h2h_a.played)

    # Step 6: Fair play — not tracked; step 7: drawing of lots
    return TiebreakerResult(winner=team_a, loser=team_b, step=7,
                            step_name=_STEP_NAMES[7],
                            margin=0, h2h_played=h2h_a.played)


# ---------------------------------------------------------------------------
# Human-readable tiebreaker context (for AI prompt + WhatsApp)
# ---------------------------------------------------------------------------

def build_tiebreaker_context(
    team_a: str,
    team_b: str,
    row_a: dict,
    row_b: dict,
    completed_matches: list[dict],
    today_is_h2h: bool = False,
) -> str:
    """
    Generate a human-readable tiebreaker scenario description for injection
    into the AI ensemble prompt and the WhatsApp context section.

    today_is_h2h: True when today's match IS team_a vs team_b (the H2H
    hasn't been played yet; this match resolves step 1 directly).
    """
    pts_a = row_a.get("points", 0)
    pts_b = row_b.get("points", 0)

    if pts_a != pts_b:
        return ""  # Not a tiebreaker situation

    if today_is_h2h:
        return (
            f"TIEBREAKER ALERT: {team_a} and {team_b} are level on {pts_a} pts. "
            f"Today's match IS the direct head-to-head tiebreaker (FIFA step 1). "
            f"The winner finishes above the loser regardless of goal difference. "
            f"A draw triggers H2H goal difference (step 2) — every goal matters."
        )

    h2h_a, h2h_b = extract_h2h(team_a, team_b, completed_matches)

    if h2h_a.played == 0:
        # H2H not yet played (and not today) — teams tied, overall GD active
        gd_a = row_a.get("goal_difference", 0)
        gd_b = row_b.get("goal_difference", 0)
        if gd_a != gd_b:
            leader = team_a if gd_a > gd_b else team_b
            trailer = team_b if leader == team_a else team_a
            margin = abs(gd_a - gd_b)
            return (
                f"TIEBREAKER: {team_a} and {team_b} level on {pts_a} pts; "
                f"H2H not yet played. Current tiebreaker: overall GD — "
                f"{leader} leads by +{margin}. "
                f"{trailer} must outscore its opponent today to close the gap."
            )
        gf_a = row_a.get("goals_for", 0)
        gf_b = row_b.get("goals_for", 0)
        return (
            f"TIEBREAKER: {team_a} and {team_b} level on pts and overall GD. "
            f"H2H not yet played. Goals scored today decide seeding (step 5): "
            f"{team_a} {gf_a} vs {team_b} {gf_b}."
        )

    # H2H already played — resolve and report
    tb = resolve_tiebreaker(team_a, team_b, row_a, row_b, completed_matches)

    if tb.step == 1:
        return (
            f"TIEBREAKER (resolved): {tb.winner} leads {tb.loser} on H2H points "
            f"(+{tb.margin} pts from their direct match). "
            f"Position effectively locked barring points change."
        )
    elif tb.step == 2:
        h2h_score = f"{h2h_a.gf}-{h2h_a.ga}"
        return (
            f"TIEBREAKER: H2H match ended {h2h_score} (draw). "
            f"{tb.winner} leads {tb.loser} on H2H goal difference by +{tb.margin}. "
            f"Step 4 (overall GD) applies if H2H GD were equal."
        )
    elif tb.step == 3:
        return (
            f"TIEBREAKER: H2H pts and GD equal. "
            f"{tb.winner} leads on H2H goals scored by +{tb.margin}."
        )
    elif tb.step == 4:
        gd_a = row_a.get("goal_difference", 0)
        gd_b = row_b.get("goal_difference", 0)
        return (
            f"TIEBREAKER: H2H fully equal. Current decider: overall group GD — "
            f"{team_a} ({gd_a:+d}) vs {team_b} ({gd_b:+d}). "
            f"{tb.winner} leads by +{tb.margin}. Today's goals could flip this."
        )
    elif tb.step == 5:
        gf_a = row_a.get("goals_for", 0)
        gf_b = row_b.get("goals_for", 0)
        return (
            f"TIEBREAKER: Overall GD equal. Decider: goals scored — "
            f"{team_a} ({gf_a}) vs {team_b} ({gf_b}). "
            f"{tb.winner} leads by +{tb.margin}. Scoring more today breaks the tie."
        )
    else:
        return (
            f"TIEBREAKER: {team_a} and {team_b} fully equal across all 5 FIFA criteria. "
            f"Would go to fair play / drawing of lots."
        )


# ---------------------------------------------------------------------------
# "8 best third-place" ranking across all 12 groups
# ---------------------------------------------------------------------------

def rank_third_place_teams(all_groups: dict) -> dict[str, int]:
    """
    Rank all third-place teams across groups for the '8 best' qualification path.

    Only groups with at least 3 teams in the data contribute.
    Ranking criteria: pts → GD → GF (FIFA steps 4–5 applied across groups).

    Returns {team_name: rank} where rank ≤ 8 = currently in qualifying position.
    A team not listed → not identifiable as 3rd in its group.
    """
    third_place: list[dict] = []

    for grp_key, rows in all_groups.items():
        ordered = sorted(
            rows,
            key=lambda r: (
                -r.get("points", 0),
                -r.get("goal_difference", 0),
                -r.get("goals_for", 0),
            ),
        )
        if len(ordered) >= 3:
            # 3rd-place team in this group
            third_place.append({**ordered[2], "_group": grp_key})

    # Sort all third-place teams: pts → GD → GF
    ranked = sorted(
        third_place,
        key=lambda r: (
            -r.get("points", 0),
            -r.get("goal_difference", 0),
            -r.get("goals_for", 0),
        ),
    )
    return {t["name"]: i + 1 for i, t in enumerate(ranked)}
