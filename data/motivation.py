"""
data/motivation.py — Tournament motivation context (rotation-trap fix).

On the final group-stage matchday, teams that have already secured
qualification — especially their seeding position — often rest key players.
Teams in elimination danger play at maximum intensity.

This module reads group_tables.json to derive a 'motivation multiplier'
(applied to λ before simulation) and builds the context block injected into
the AI prompt and WhatsApp message.

Public API:
    load_group_tables(path) -> dict
    get_team_entry(team_name, tables) -> dict | None
    motivation_multiplier(status) -> float
    build_match_motivation(home_team, away_team, tables) -> MatchMotivation
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Motivation multipliers — applied to blended λ *before* simulation
# ---------------------------------------------------------------------------

MOTIVATION_MULTIPLIER: dict[str, float] = {
    "qualified_secure_1st":   0.85,  # 1st place mathematically locked — heavy rotation
    "qualified":              0.92,  # qualified & seeding settled (2nd locked) — minor rotation
    "qualified_top_seed_fight": 1.05, # qualified but 1st vs 2nd seed still contested — full squad
    "need_draw":              1.05,  # draw qualifies — high motivation, patient play
    "must_win":               1.10,  # win-or-go-home — maximum intensity
    "open":                   1.00,  # qualification still open — normal
    "eliminated":             0.90,  # pride only — possible youth appearances
    "unknown":                1.00,  # no table data — no adjustment
}

_CONTEXT_LABEL: dict[str, str] = {
    "qualified_secure_1st":     "Already qualified & 1st place locked in — heavy rotation expected",
    "qualified":                "Already qualified, seeding settled — likely to rotate some players",
    "qualified_top_seed_fight": "Qualified but fighting for top seed — full-strength squad expected",
    "need_draw":                "A draw qualifies — high motivation",
    "must_win":                 "Must win to stay in tournament — full-strength squad expected",
    "open":                     "Qualification still open — full motivation",
    "eliminated":               "Already eliminated — playing for pride only",
    "unknown":                  "",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TeamMotivation:
    team_name:            str
    group:                Optional[str]
    position:             int           # current group standing (1–4)
    points:               int
    played:               int
    qualification_status: str
    lambda_multiplier:    float         # λ modifier to apply before simulation
    context_label:        str           # human-readable label for WhatsApp/AI


@dataclass
class MatchMotivation:
    home: TeamMotivation
    away: TeamMotivation

    def is_trivial(self) -> bool:
        """True when neither team has a motivation adjustment."""
        return (
            self.home.qualification_status in ("open", "unknown") and
            self.away.qualification_status in ("open", "unknown")
        )

    def has_rotation_risk(self) -> bool:
        return self.home.qualification_status in ("qualified_secure_1st", "qualified") or \
               self.away.qualification_status in ("qualified_secure_1st", "qualified")

    def to_ai_section(self) -> str:
        """Prompt block prepended to the AI's context_section."""
        if self.is_trivial():
            return ""
        lines = ["TOURNAMENT CONTEXT (group-stage final matchday):"]
        for tm in (self.home, self.away):
            if tm.context_label:
                lines.append(f"  {tm.team_name}: {tm.context_label}.")
        lines.append(
            f"NOTE: Expected goals already adjusted by motivation multiplier "
            f"(home ×{self.home.lambda_multiplier:.2f}, away ×{self.away.lambda_multiplier:.2f})."
        )
        if self.has_rotation_risk():
            lines.append(
                "⚠️ ROTATION RISK DETECTED: Your reasoning MUST explicitly state which team "
                "is rotating squad and how this affects your exact-score prediction."
            )
        return "\n".join(lines)

    def to_whatsapp_lines(self) -> list[str]:
        """Lines appended to the WhatsApp match block."""
        home_lbl = self.home.context_label
        away_lbl = self.away.context_label
        if not home_lbl and not away_lbl:
            return []
        out = ["   ⚠️ *Contexto do Torneio:*"]
        if home_lbl:
            out.append(f"      {self.home.team_name}: {home_lbl}")
        if away_lbl:
            out.append(f"      {self.away.team_name}: {away_lbl}")
        return out


# ---------------------------------------------------------------------------
# Name normalisation (mirrors winner_odds_loader._norm to avoid cross-import)
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    name = name.strip()
    if name and not name[0].isascii():
        parts = name.split(None, 1)
        name = parts[1] if len(parts) > 1 else ""
    return name.lower().strip()


def _teams_match(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    return na == nb or na in nb or nb in na


# ---------------------------------------------------------------------------
# Qualification status computation (from raw group rows)
# ---------------------------------------------------------------------------

def compute_group_statuses(rows: list[dict]) -> list[dict]:
    """
    Derive qualification_status for each team from their current standing.

    Mutates each dict in-place and returns the sorted list (1st → 4th).
    Uses a conservative heuristic suited for WC 2026 matchday 3.

    Status values:
        qualified_secure_1st — already locked into 1st (heavy rotation likely)
        qualified            — top-2 guaranteed regardless of today's result
        need_draw            — a draw is enough to qualify
        must_win             — must win; hope for other result to go their way
        eliminated           — cannot finish top-2 even with a win
        open                 — still contested, any result possible
    """
    # Sort: points desc → goal diff desc → goals for desc
    ordered = sorted(
        rows,
        key=lambda r: (
            -r.get("points", 0),
            -r.get("goal_difference", 0),
            -r.get("goals_for", 0),
        ),
    )

    for pos_0, row in enumerate(ordered):
        played    = row.get("played", 0)
        pts       = row.get("points", 0)
        remaining = max(0, 3 - played)
        all_pts   = [r.get("points", 0) for r in ordered]

        if remaining == 0:
            # All games played — position is final
            row["qualification_status"] = "qualified" if pos_0 < 2 else "eliminated"
            continue

        if played < 2:
            # Matchday 1 or 2 — too early to determine
            row["qualification_status"] = "open"
            continue

        # ── Matchday 3: one game left ──────────────────────────────────────
        # Can the team 3 places below still reach this team's points?
        third_pts = all_pts[2] if len(all_pts) > 2 else 0
        fourth_pts = all_pts[3] if len(all_pts) > 3 else 0

        if pos_0 == 0:       # Currently 1st
            # Secure 1st if 2nd-place can't catch us even with a win
            if pts > (all_pts[1] + remaining * 3):
                # 1st place mathematically locked — heavy rotation expected
                row["qualification_status"] = "qualified_secure_1st"
            elif pts > third_pts + remaining * 3:
                # Can't drop below 2nd — qualification guaranteed.
                # But 1st place is NOT yet locked → fighting for top seed.
                row["qualification_status"] = "qualified_top_seed_fight"
            else:
                # 3rd could overtake → qualification not yet certain → full effort
                row["qualification_status"] = "open"

        elif pos_0 == 1:     # Currently 2nd
            third_max = third_pts + remaining * 3
            if pts > third_max:
                # 3rd can't catch us — qualified.
                # Now check whether winning today could leapfrog current 1st.
                if pts + 3 >= all_pts[0]:
                    # A win could equal or beat 1st place's current points →
                    # seeding still contested → play for the top seed.
                    row["qualification_status"] = "qualified_top_seed_fight"
                else:
                    # Can't reach 1st regardless — seeding settled → minor rotation OK
                    row["qualification_status"] = "qualified"
            elif pts == third_max:
                # 3rd reaches our current points only if they win AND we lose
                row["qualification_status"] = "need_draw"
            else:
                # 3rd can actually overtake us by winning while we draw/lose
                row["qualification_status"] = "need_draw" if pts >= 3 else "must_win"

        elif pos_0 == 2:     # Currently 3rd
            second_pts = all_pts[1]
            if pts + remaining * 3 < second_pts:
                row["qualification_status"] = "eliminated"
            else:
                row["qualification_status"] = "must_win"

        else:                # Currently 4th (or lower)
            first_pts = all_pts[0]
            if pts + remaining * 3 < first_pts:
                row["qualification_status"] = "eliminated"
            else:
                row["qualification_status"] = "must_win"

    return ordered


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_group_tables(path: str = "data/group_tables.json") -> dict:
    """
    Load group_tables.json.  Returns {} if absent (no crash — pipeline continues).
    """
    try:
        with open(path, encoding="utf-8") as f:
            tables = json.load(f)
        n_groups = len(tables.get("groups", {}))
        print(f"[motivation] Loaded group tables: {n_groups} group(s) from '{path}'.")
        return tables
    except FileNotFoundError:
        print(f"[motivation] '{path}' not found — motivation adjustment skipped.")
        return {}


def get_team_entry(team_name: str, tables: dict) -> Optional[dict]:
    """Find a team's row (with injected 'group' key) by fuzzy name match."""
    for grp_key, grp_teams in tables.get("groups", {}).items():
        for entry in grp_teams:
            if _teams_match(team_name, entry.get("name", "")):
                return {**entry, "group": grp_key}
    return None


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_match_motivation(
    home_team: str,
    away_team: str,
    tables: dict,
) -> MatchMotivation:
    """
    Build MatchMotivation for a given match from the loaded group tables.

    When no table data is available for a team, motivation multiplier = 1.0
    and context_label = "" — so the pipeline continues normally.
    """
    def _make(team_name: str) -> TeamMotivation:
        entry = get_team_entry(team_name, tables)
        if entry is None:
            return TeamMotivation(
                team_name=team_name, group=None, position=0,
                points=0, played=0,
                qualification_status="unknown",
                lambda_multiplier=1.0,
                context_label="",
            )
        status = entry.get("qualification_status", "unknown")
        return TeamMotivation(
            team_name          = team_name,
            group              = entry.get("group"),
            position           = entry.get("position", 0),
            points             = entry.get("points", 0),
            played             = entry.get("played", 0),
            qualification_status = status,
            lambda_multiplier  = MOTIVATION_MULTIPLIER.get(status, 1.0),
            context_label      = _CONTEXT_LABEL.get(status, ""),
        )

    return MatchMotivation(home=_make(home_team), away=_make(away_team))
