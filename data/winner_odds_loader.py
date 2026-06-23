"""
data/winner_odds_loader.py — Load winner_odds.json and match to morning_picks.

Supports two file formats:

  OLD (flat list):
    [{"home_team": "Spain", "away_team": "Saudi Arabia",
      "odds_home": 1.25, "odds_draw": 6.50, "odds_away": 13.00}, ...]

  NEW (dict with sub-markets):
    {
      "Spain vs Saudi Arabia": {
        "winner":      {"home": 1.25, "draw": 6.50, "away": 13.00},
        "sum_goals":   {"0-1": 4.50, "2-3": 2.10, "+4": 3.20},
        "corners_range": {"0-8": 2.20, "9-11": 2.55, "12+": 3.10}
      },
      ...
    }
  Keys starting with "_" (e.g. "_note") are skipped.

Decimal odds only (e.g. 1.25, 6.50).
Team names can be plain ("Spain") or flag-prefixed ("🇪🇸 Spain") — both work.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class MatchEV:
    home_team: str
    away_team: str
    # 1X2 odds from winner_odds.json
    odds_home: float
    odds_draw: float
    odds_away: float
    # Sum-goals bracket book odds (None if not in file)
    sg_01_odds:    Optional[float]   # odds for 0-1 total goals
    sg_23_odds:    Optional[float]   # odds for 2-3 total goals
    sg_4plus_odds: Optional[float]   # odds for 4+ total goals
    # AI sim probabilities from morning_picks (None if not yet run)
    sim_p_home: Optional[float]
    sim_p_draw: Optional[float]
    sim_p_away: Optional[float]
    # EV = probability × decimal_odds − 1  (None if inputs missing)
    ev_home: Optional[float]
    ev_draw: Optional[float]
    ev_away: Optional[float]
    # Best EV across the three 1X2 outcomes
    ev_winner: Optional[float]
    ev_winner_outcome: Optional[str]   # "home" | "draw" | "away"
    # Sum-goals bracket EV (None if book odds or model prob missing)
    ev_sg_01:    Optional[float]
    ev_sg_23:    Optional[float]
    ev_sg_4plus: Optional[float]
    matched: bool                      # False = no odds entry found


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """
    Strip leading emoji-flag token, lowercase, strip whitespace.

    "🇪🇸 Spain"   → "spain"
    "Spain"         → "spain"
    "Saudi Arabia"  → "saudi arabia"
    "IR Iran"       → "ir iran"   (substring match will still catch "Iran")
    """
    name = name.strip()
    if name and not name[0].isascii():          # leading emoji flag present
        parts = name.split(None, 1)
        name = parts[1] if len(parts) > 1 else ""
    return name.lower().strip()


def _teams_match(picks_name: str, odds_name: str) -> bool:
    """True when both names refer to the same team after normalisation."""
    p = _norm(picks_name)
    o = _norm(odds_name)
    return p == o or p in o or o in p


# ---------------------------------------------------------------------------
# Internal: normalise both file formats into a consistent list
# ---------------------------------------------------------------------------

def _load_odds_file(path: str) -> list[dict]:
    """
    Load winner_odds.json in either format and return a normalised list.

    Each element:
    {
        "home_team":    str,
        "away_team":    str,
        "odds_home":    float,
        "odds_draw":    float,
        "odds_away":    float,
        "sg_01":        float | None,   # book odds for 0-1 goals
        "sg_23":        float | None,   # book odds for 2-3 goals
        "sg_4plus":     float | None,   # book odds for 4+ goals
        "corners_range": dict | None,
    }

    Raises FileNotFoundError if path doesn't exist.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    normalised: list[dict] = []

    if isinstance(raw, list):
        # ── OLD flat-list format ──────────────────────────────────────────
        for entry in raw:
            normalised.append({
                "home_team":    entry.get("home_team", ""),
                "away_team":    entry.get("away_team", ""),
                "odds_home":    float(entry.get("odds_home") or 0),
                "odds_draw":    float(entry.get("odds_draw") or 0),
                "odds_away":    float(entry.get("odds_away") or 0),
                "sg_01":        None,
                "sg_23":        None,
                "sg_4plus":     None,
                "corners_range": None,
            })

    elif isinstance(raw, dict):
        # ── NEW dict format: {"Team A vs Team B": {winner:..., sum_goals:...}} ─
        for key, sub in raw.items():
            if key.startswith("_"):      # skip metadata keys like "_note"
                continue
            if not isinstance(sub, dict):
                continue

            # Parse "Team A vs Team B" key
            if " vs " in key:
                home_str, away_str = key.split(" vs ", 1)
            else:
                # Fallback: try alternative separators
                home_str, away_str = key, ""

            winner_odds = sub.get("winner", {}) or {}
            sg          = sub.get("sum_goals", {}) or {}

            normalised.append({
                "home_team":    home_str.strip(),
                "away_team":    away_str.strip(),
                "odds_home":    float(winner_odds.get("home") or 0),
                "odds_draw":    float(winner_odds.get("draw") or 0),
                "odds_away":    float(winner_odds.get("away") or 0),
                "sg_01":        float(sg.get("0-1") or 0) or None,
                "sg_23":        float(sg.get("2-3") or 0) or None,
                "sg_4plus":     float(sg.get("+4")  or 0) or None,
                "corners_range": sub.get("corners_range"),
            })

    return normalised


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_odds(odds_path: str = "winner_odds.json") -> list[dict]:
    """
    Load and normalise winner_odds.json. Returns [] if file absent.
    Call once before the match loop; pass the result to find_match_odds().
    """
    try:
        data = _load_odds_file(odds_path)
        print(f"[winner_odds] Loaded {len(data)} entry/entries from '{odds_path}'.")
        return data
    except FileNotFoundError:
        print(f"[winner_odds] '{odds_path}' not found — O/U EV detection skipped.")
        return []


def find_match_odds(
    home_team: str,
    away_team: str,
    odds_list: list[dict],
) -> Optional[dict]:
    """
    Find the normalised odds entry for a given match.
    Returns the entry dict (or a home/away-swapped copy) if found, else None.
    """
    for entry in odds_list:
        oh = entry.get("home_team", "")
        oa = entry.get("away_team", "")

        if _teams_match(home_team, oh) and _teams_match(away_team, oa):
            return entry

        # Odds file lists teams in reverse order → swap winner odds only
        if _teams_match(home_team, oa) and _teams_match(away_team, oh):
            return {
                **entry,
                "home_team":  home_team,
                "away_team":  away_team,
                "odds_home":  entry["odds_away"],
                "odds_away":  entry["odds_home"],
            }

    return None


def load_and_match(
    odds_path: str  = "winner_odds.json",
    picks_path: str = "data/morning_picks.json",
) -> list[MatchEV]:
    """
    Match every entry in morning_picks.json to winner_odds.json by team name.
    Returns one MatchEV per pick (matched=False if no odds entry found).

    EV formula: EV = sim_probability × decimal_odds − 1
      EV > 0  → positive expected value (value bet)
      EV ≤ 0  → bookmaker has the edge
    """
    # -- load odds ----------------------------------------------------------
    try:
        odds_list = _load_odds_file(odds_path)
        print(f"[winner_odds] Loaded {len(odds_list)} odds entry/entries from '{odds_path}'.")
    except FileNotFoundError:
        print(f"[winner_odds] '{odds_path}' not found — create it first (see module docstring).")
        return []

    # -- load picks ---------------------------------------------------------
    try:
        with open(picks_path, encoding="utf-8") as f:
            picks_list: list[dict] = json.load(f)
        print(f"[winner_odds] Loaded {len(picks_list)} pick(s) from '{picks_path}'.")
    except FileNotFoundError:
        print(f"[winner_odds] '{picks_path}' not found — run the morning pipeline first.")
        return []

    results: list[MatchEV] = []

    for pick in picks_list:
        ph = pick.get("home_team", "")
        pa = pick.get("away_team", "")

        entry = find_match_odds(ph, pa, odds_list)

        if entry is None:
            print(f"[winner_odds]   NO MATCH: '{ph}' vs '{pa}'")
            results.append(MatchEV(
                home_team=ph, away_team=pa,
                odds_home=0.0, odds_draw=0.0, odds_away=0.0,
                sg_01_odds=None, sg_23_odds=None, sg_4plus_odds=None,
                sim_p_home=None, sim_p_draw=None, sim_p_away=None,
                ev_home=None, ev_draw=None, ev_away=None,
                ev_winner=None, ev_winner_outcome=None,
                ev_sg_01=None, ev_sg_23=None, ev_sg_4plus=None,
                matched=False,
            ))
            continue

        odds_h = entry["odds_home"]
        odds_d = entry["odds_draw"]
        odds_a = entry["odds_away"]
        sg_01    = entry.get("sg_01")
        sg_23    = entry.get("sg_23")
        sg_4plus = entry.get("sg_4plus")

        sim_h       = pick.get("sim_p_home")
        sim_d       = pick.get("sim_p_draw")
        sim_a       = pick.get("sim_p_away")
        model_sg_01    = pick.get("model_sg_01")
        model_sg_23    = pick.get("model_sg_23")
        model_sg_4plus = pick.get("model_sg_4plus")

        # EV = p × odds − 1
        ev_h = round(sim_h * odds_h - 1, 4) if (sim_h is not None and odds_h) else None
        ev_d = round(sim_d * odds_d - 1, 4) if (sim_d is not None and odds_d) else None
        ev_a = round(sim_a * odds_a - 1, 4) if (sim_a is not None and odds_a) else None

        ev_sg_01    = round(model_sg_01    * sg_01    - 1, 4) if (model_sg_01    and sg_01)    else None
        ev_sg_23    = round(model_sg_23    * sg_23    - 1, 4) if (model_sg_23    and sg_23)    else None
        ev_sg_4plus = round(model_sg_4plus * sg_4plus - 1, 4) if (model_sg_4plus and sg_4plus) else None

        # best 1X2 EV outcome
        candidates = [(ev, lbl) for ev, lbl in [(ev_h, "home"), (ev_d, "draw"), (ev_a, "away")]
                      if ev is not None]
        best_ev, best_lbl = max(candidates, key=lambda x: x[0]) if candidates else (None, None)

        if ev_h is not None:
            print(
                f"[winner_odds]   MATCHED: '{ph}' vs '{pa}' | "
                f"odds H={odds_h} D={odds_d} A={odds_a} | "
                f"EV H={ev_h:+.2%}  D={ev_d:+.2%}  A={ev_a:+.2%}  "
                f"-> best={best_lbl}({best_ev:+.2%})"
            )
        else:
            print(
                f"[winner_odds]   MATCHED (no sim probs yet): '{ph}' vs '{pa}' | "
                f"odds H={odds_h} D={odds_d} A={odds_a}"
            )

        if ev_sg_01 is not None:
            print(
                f"[winner_odds]   Sum Goals EV: 0-1={ev_sg_01:+.2%}  2-3={ev_sg_23:+.2%}  4+={ev_sg_4plus:+.2%}"
            )

        results.append(MatchEV(
            home_team=ph, away_team=pa,
            odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
            sg_01_odds=sg_01, sg_23_odds=sg_23, sg_4plus_odds=sg_4plus,
            sim_p_home=sim_h, sim_p_draw=sim_d, sim_p_away=sim_a,
            ev_home=ev_h, ev_draw=ev_d, ev_away=ev_a,
            ev_winner=best_ev, ev_winner_outcome=best_lbl,
            ev_sg_01=ev_sg_01, ev_sg_23=ev_sg_23, ev_sg_4plus=ev_sg_4plus,
            matched=True,
        ))

    return results


def enrich_picks(
    morning_data: list[dict],
    odds_path: str = "winner_odds.json",
) -> list[dict]:
    """
    Enrich morning_data dicts (in memory) with EV fields from winner_odds.json.
    Mutates each dict in-place; returns the same list.

    Adds the following fields to each dict (None when data unavailable):
      ev_home, ev_draw, ev_away       — 1X2 expected value
      ev_winner, ev_winner_outcome    — best 1X2 EV
      ev_sg_01, ev_sg_23, ev_sg_4plus — sum-goals bracket EV (needs model_sg_01/23/4plus)

    Called by main.py after the Step-4 loop, before save_morning_picks().
    Gracefully no-ops when winner_odds.json is absent — pipeline continues unchanged.
    """
    try:
        odds_list = _load_odds_file(odds_path)
        print(f"[winner_odds] Loaded {len(odds_list)} entry/entries from '{odds_path}'.")
    except FileNotFoundError:
        print(f"[winner_odds] '{odds_path}' not found — skipping EV enrichment.")
        return morning_data

    for pick in morning_data:
        ph = pick.get("home_team", "")
        pa = pick.get("away_team", "")

        entry = find_match_odds(ph, pa, odds_list)

        if entry is None:
            print(f"[winner_odds] No odds match for '{ph}' vs '{pa}' — EV fields set to None.")
            pick.update({
                "ev_home": None, "ev_draw": None, "ev_away": None,
                "ev_winner": None, "ev_winner_outcome": None,
                "ev_sg_01": None, "ev_sg_23": None, "ev_sg_4plus": None,
            })
            continue

        odds_h = entry["odds_home"]
        odds_d = entry["odds_draw"]
        odds_a = entry["odds_away"]
        sg_01    = entry.get("sg_01")
        sg_23    = entry.get("sg_23")
        sg_4plus = entry.get("sg_4plus")

        sim_h          = pick.get("sim_p_home")
        sim_d          = pick.get("sim_p_draw")
        sim_a          = pick.get("sim_p_away")
        model_sg_01    = pick.get("model_sg_01")
        model_sg_23    = pick.get("model_sg_23")
        model_sg_4plus = pick.get("model_sg_4plus")

        ev_h = round(sim_h * odds_h - 1, 4) if (sim_h is not None and odds_h) else None
        ev_d = round(sim_d * odds_d - 1, 4) if (sim_d is not None and odds_d) else None
        ev_a = round(sim_a * odds_a - 1, 4) if (sim_a is not None and odds_a) else None

        ev_sg_01    = round(model_sg_01    * sg_01    - 1, 4) if (model_sg_01    and sg_01)    else None
        ev_sg_23    = round(model_sg_23    * sg_23    - 1, 4) if (model_sg_23    and sg_23)    else None
        ev_sg_4plus = round(model_sg_4plus * sg_4plus - 1, 4) if (model_sg_4plus and sg_4plus) else None

        candidates = [(ev, lbl) for ev, lbl in
                      [(ev_h, "home"), (ev_d, "draw"), (ev_a, "away")] if ev is not None]
        best_ev, best_lbl = max(candidates, key=lambda x: x[0]) if candidates else (None, None)

        if ev_h is not None:
            print(
                f"[winner_odds] EV '{ph}' vs '{pa}': "
                f"H={ev_h:+.2%}  D={ev_d:+.2%}  A={ev_a:+.2%}  -> best={best_lbl}({best_ev:+.2%})"
            )
        else:
            print(f"[winner_odds] Matched (no sim probs yet): '{ph}' vs '{pa}'")

        if ev_sg_01 is not None:
            print(
                f"[winner_odds]   Sum Goals EV: 0-1={ev_sg_01:+.2%}  2-3={ev_sg_23:+.2%}  4+={ev_sg_4plus:+.2%}"
            )

        pick.update({
            "ev_home":           ev_h,
            "ev_draw":           ev_d,
            "ev_away":           ev_a,
            "ev_winner":         best_ev,
            "ev_winner_outcome": best_lbl,
            "ev_sg_01":          ev_sg_01,
            "ev_sg_23":          ev_sg_23,
            "ev_sg_4plus":       ev_sg_4plus,
        })

    return morning_data
