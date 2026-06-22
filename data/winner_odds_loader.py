"""
data/winner_odds_loader.py — Load winner_odds.json and match to morning_picks.

Standalone helper — does NOT modify any files.
Call load_and_match() to get a list of MatchEV records you can inspect or
pass to the pipeline for persisting ev_winner back into morning_picks.json.

Expected winner_odds.json format (place in project root):
[
  {
    "home_team": "Spain",
    "away_team": "Saudi Arabia",
    "odds_home": 1.25,
    "odds_draw": 6.50,
    "odds_away": 13.00
  },
  ...
]

Decimal odds only (e.g. 1.25 = "4/1 on", 6.50 = "11/2").
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
    # Odds from winner_odds.json
    odds_home: float
    odds_draw: float
    odds_away: float
    # AI sim probabilities from morning_picks (None if not yet run)
    sim_p_home: Optional[float]
    sim_p_draw: Optional[float]
    sim_p_away: Optional[float]
    # EV = probability × decimal_odds − 1  (None if inputs missing)
    ev_home: Optional[float]
    ev_draw: Optional[float]
    ev_away: Optional[float]
    # Best EV across the three outcomes
    ev_winner: Optional[float]
    ev_winner_outcome: Optional[str]   # "home" | "draw" | "away"
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
# Public API
# ---------------------------------------------------------------------------

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
        with open(odds_path, encoding="utf-8") as f:
            odds_list: list[dict] = json.load(f)
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

        # -- find matching odds entry --------------------------------------
        matched_odds: dict | None = None
        swapped = False

        for entry in odds_list:
            oh = entry.get("home_team", "")
            oa = entry.get("away_team", "")

            if _teams_match(ph, oh) and _teams_match(pa, oa):
                matched_odds = entry
                break

            # Odds file sometimes lists teams in reverse order
            if _teams_match(ph, oa) and _teams_match(pa, oh):
                matched_odds = {
                    **entry,
                    "odds_home": entry.get("odds_away"),
                    "odds_away": entry.get("odds_home"),
                }
                swapped = True
                break

        if matched_odds is None:
            print(f"[winner_odds]   NO MATCH: '{ph}' vs '{pa}'")
            results.append(MatchEV(
                home_team=ph, away_team=pa,
                odds_home=0.0, odds_draw=0.0, odds_away=0.0,
                sim_p_home=None, sim_p_draw=None, sim_p_away=None,
                ev_home=None, ev_draw=None, ev_away=None,
                ev_winner=None, ev_winner_outcome=None,
                matched=False,
            ))
            continue

        if swapped:
            print(f"[winner_odds]   NOTE: home/away swapped in odds file for '{ph}' vs '{pa}'")

        odds_h = float(matched_odds.get("odds_home") or 0)
        odds_d = float(matched_odds.get("odds_draw") or 0)
        odds_a = float(matched_odds.get("odds_away") or 0)

        sim_h = pick.get("sim_p_home")
        sim_d = pick.get("sim_p_draw")
        sim_a = pick.get("sim_p_away")

        # EV = p × odds − 1
        ev_h = round(sim_h * odds_h - 1, 4) if (sim_h is not None and odds_h) else None
        ev_d = round(sim_d * odds_d - 1, 4) if (sim_d is not None and odds_d) else None
        ev_a = round(sim_a * odds_a - 1, 4) if (sim_a is not None and odds_a) else None

        # best EV outcome
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

        results.append(MatchEV(
            home_team=ph, away_team=pa,
            odds_home=odds_h, odds_draw=odds_d, odds_away=odds_a,
            sim_p_home=sim_h, sim_p_draw=sim_d, sim_p_away=sim_a,
            ev_home=ev_h, ev_draw=ev_d, ev_away=ev_a,
            ev_winner=best_ev,
            ev_winner_outcome=best_lbl,
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

    Called by main.py after the Step-4 loop, before save_morning_picks().
    Gracefully no-ops when winner_odds.json is absent — pipeline continues unchanged.
    """
    try:
        with open(odds_path, encoding="utf-8") as f:
            odds_list: list[dict] = json.load(f)
        print(f"[winner_odds] Loaded {len(odds_list)} entry/entries from '{odds_path}'.")
    except FileNotFoundError:
        print(f"[winner_odds] '{odds_path}' not found — skipping EV enrichment.")
        return morning_data

    for pick in morning_data:
        ph = pick.get("home_team", "")
        pa = pick.get("away_team", "")

        matched_odds: dict | None = None
        swapped = False

        for entry in odds_list:
            oh = entry.get("home_team", "")
            oa = entry.get("away_team", "")
            if _teams_match(ph, oh) and _teams_match(pa, oa):
                matched_odds = entry
                break
            if _teams_match(ph, oa) and _teams_match(pa, oh):
                matched_odds = {**entry, "odds_home": entry.get("odds_away"),
                                          "odds_away": entry.get("odds_home")}
                swapped = True
                break

        if matched_odds is None:
            print(f"[winner_odds] No odds match for '{ph}' vs '{pa}' — EV fields set to None.")
            pick.update({"ev_home": None, "ev_draw": None, "ev_away": None,
                         "ev_winner": None, "ev_winner_outcome": None})
            continue

        if swapped:
            print(f"[winner_odds]   NOTE: home/away swapped for '{ph}' vs '{pa}'")

        odds_h = float(matched_odds.get("odds_home") or 0)
        odds_d = float(matched_odds.get("odds_draw") or 0)
        odds_a = float(matched_odds.get("odds_away") or 0)

        sim_h = pick.get("sim_p_home")
        sim_d = pick.get("sim_p_draw")
        sim_a = pick.get("sim_p_away")

        ev_h = round(sim_h * odds_h - 1, 4) if (sim_h is not None and odds_h) else None
        ev_d = round(sim_d * odds_d - 1, 4) if (sim_d is not None and odds_d) else None
        ev_a = round(sim_a * odds_a - 1, 4) if (sim_a is not None and odds_a) else None

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

        pick.update({
            "ev_home":           ev_h,
            "ev_draw":           ev_d,
            "ev_away":           ev_a,
            "ev_winner":         best_ev,
            "ev_winner_outcome": best_lbl,
        })

    return morning_data
