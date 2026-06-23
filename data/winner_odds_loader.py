"""
data/winner_odds_loader.py — Load winner_odds.json and match to morning_picks.

Supports two file formats:

  OLD (flat list):
    [{"home_team": "Spain", "away_team": "Saudi Arabia",
      "odds_home": 1.25, "odds_draw": 6.50, "odds_away": 13.00}, ...]

  NEW (dict with sub-markets):
    {
      "Spain vs Saudi Arabia": {
        "winner":         {"home": 1.25, "draw": 6.50, "away": 13.00},
        "over_under_2_5": {"over": 1.80, "under": 2.00},
        "corners_range":  {"0-8": 2.20, "9-11": 2.55, "12+": 3.10}
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
    # O/U 2.5 book odds (None if not in file)
    ou25_over:  Optional[float]
    ou25_under: Optional[float]
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
    # O/U 2.5 EV (None if book odds or model prob missing)
    ev_ou_over:  Optional[float]
    ev_ou_under: Optional[float]
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
        "home_team":  str,
        "away_team":  str,
        "odds_home":  float,
        "odds_draw":  float,
        "odds_away":  float,
        "ou25_over":  float | None,
        "ou25_under": float | None,
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
                "ou25_over":    None,
                "ou25_under":   None,
                "corners_range": None,
            })

    elif isinstance(raw, dict):
        # ── NEW dict format: {"Team A vs Team B": {winner:..., ou25:...}} ─
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
            ou25        = sub.get("over_under_2_5", {}) or {}

            normalised.append({
                "home_team":    home_str.strip(),
                "away_team":    away_str.strip(),
                "odds_home":    float(winner_odds.get("home") or 0),
                "odds_draw":    float(winner_odds.get("draw") or 0),
                "odds_away":    float(winner_odds.get("away") or 0),
                "ou25_over":    float(ou25.get("over") or 0) or None,
                "ou25_under":   float(ou25.get("under") or 0) or None,
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
                ou25_over=None, ou25_under=None,
                sim_p_home=None, sim_p_draw=None, sim_p_away=None,
                ev_home=None, ev_draw=None, ev_away=None,
                ev_winner=None, ev_winner_outcome=None,
                ev_ou_over=None, ev_ou_under=None,
                matched=False,
            ))
            continue

        odds_h = entry["odds_home"]
        odds_d = entry["odds_draw"]
        odds_a = entry["odds_away"]
        ou_over  = entry.get("ou25_over")
        ou_under = entry.get("ou25_under")

        sim_h = pick.get("sim_p_home")
        sim_d = pick.get("sim_p_draw")
        sim_a = pick.get("sim_p_away")
        model_over = pick.get("model_p_over_2_5")

        # EV = p × odds − 1
        ev_h = round(sim_h * odds_h - 1, 4) if (sim_h is not None and odds_h) else None
        ev_d = round(sim_d * odds_d - 1, 4) if (sim_d is not None and odds_d) else None
        ev_a = round(sim_a * odds_a - 1, 4) if (sim_a is not None and odds_a) else None

        ev_ou_over  = round(model_over * ou_over - 1, 4) \
            if (model_over is not None and ou_over) else None
        ev_ou_under = round((1 - model_over) * ou_under - 1, 4) \
            if (model_over is not None and ou_under) else None

        # best 1X2 EV outcome
        candidates = [(ev, lbl) for ev, lbl in [(ev_h, "home"), (ev_d, "draw"), (ev_a, "away")]
                      if ev is not None]
        best_ev, best_lbl = max(candidates, key=lambda x: x[0]) if candidates else (None, None)

        if ev_h is not None:
            print(
                f"[winner_odds]   MATCHED: '{ph}' vs '{pa}' | "
                f"odds H={odds_h} D={odds_d} A={odds_a} | "
                f"EV H={ev_h:+.2%}  D={ev_d:+.2%}  A={ev_a:+.2%}  "
                f"→ best={best_lbl}({best_ev:+.2%})"
            )
        else:
            print(
                f"[winner_odds]   MATCHED (no sim probs yet): '{ph}' vs '{pa}' | "
                f"odds H={odds_h} D={odds_d} A={odds_a}"
            )

        if ev_ou_over is not None:
            print(
                f"[winner_odds]   O/U 2.5: over={ev_ou_over:+.2%}  under={ev_ou_under:+.2%}"
            )

        results.append(MatchEV(
            home_team=ph, away_team=pa,
            odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
            ou25_over=ou_over, ou25_under=ou_under,
            sim_p_home=sim_h, sim_p_draw=sim_d, sim_p_away=sim_a,
            ev_home=ev_h, ev_draw=ev_d, ev_away=ev_a,
            ev_winner=best_ev, ev_winner_outcome=best_lbl,
            ev_ou_over=ev_ou_over, ev_ou_under=ev_ou_under,
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
      ev_ou_over, ev_ou_under         — O/U 2.5 expected value (needs model_p_over_2_5)

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
                "ev_ou_over": None, "ev_ou_under": None,
            })
            continue

        odds_h = entry["odds_home"]
        odds_d = entry["odds_draw"]
        odds_a = entry["odds_away"]
        ou_over  = entry.get("ou25_over")
        ou_under = entry.get("ou25_under")

        sim_h  = pick.get("sim_p_home")
        sim_d  = pick.get("sim_p_draw")
        sim_a  = pick.get("sim_p_away")
        model_over = pick.get("model_p_over_2_5")

        ev_h = round(sim_h * odds_h - 1, 4) if (sim_h is not None and odds_h) else None
        ev_d = round(sim_d * odds_d - 1, 4) if (sim_d is not None and odds_d) else None
        ev_a = round(sim_a * odds_a - 1, 4) if (sim_a is not None and odds_a) else None

        ev_ou_over  = round(model_over * ou_over - 1, 4) \
            if (model_over is not None and ou_over) else None
        ev_ou_under = round((1 - model_over) * ou_under - 1, 4) \
            if (model_over is not None and ou_under) else None

        candidates = [(ev, lbl) for ev, lbl in
                      [(ev_h, "home"), (ev_d, "draw"), (ev_a, "away")] if ev is not None]
        best_ev, best_lbl = max(candidates, key=lambda x: x[0]) if candidates else (None, None)

        if ev_h is not None:
            print(
                f"[winner_odds] EV '{ph}' vs '{pa}': "
                f"H={ev_h:+.2%}  D={ev_d:+.2%}  A={ev_a:+.2%}  → best={best_lbl}({best_ev:+.2%})"
            )
        else:
            print(f"[winner_odds] Matched (no sim probs yet): '{ph}' vs '{pa}'")

        if ev_ou_over is not None:
            print(
                f"[winner_odds]   O/U 2.5 EV: over={ev_ou_over:+.2%}  under={ev_ou_under:+.2%}"
            )

        pick.update({
            "ev_home":           ev_h,
            "ev_draw":           ev_d,
            "ev_away":           ev_a,
            "ev_winner":         best_ev,
            "ev_winner_outcome": best_lbl,
            "ev_ou_over":        ev_ou_over,
            "ev_ou_under":       ev_ou_under,
        })

    return morning_data
