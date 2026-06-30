"""
data/opta_priors.py — Opta Supercomputer tournament probabilities.

Provides three integration points for the mundial_predictor pipeline:

1. Context injection: per-team Opta tournament win% → Claude prompt
2. Sentiment gap detection: Opta-implied match strength vs our Poisson model
3. Statistical tiebreaker: break near-equal simulation probabilities using Opta strength ratio

Data source: theanalyst.com — 25,000-simulation Opta supercomputer.
Update opta_priors.json after each major round from the Opta article.
"""
from __future__ import annotations

import json
import os
import unicodedata
from typing import Optional

_PRIORS_PATH = os.path.join(os.path.dirname(__file__), "opta_priors.json")

# Gap threshold: flag to Claude when our model deviates > this from Opta-implied strength
SENTIMENT_GAP_THRESHOLD = 0.10   # 10 pp — tightened from 15pp after Germany/Netherlands misses

# Tiebreaker threshold: if |p_home - p_away| < this, Opta breaks the tie
TIEBREAKER_THRESHOLD = 0.02       # 2 pp — effectively equal simulation output


# ── Helpers ──────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Lowercase, strip diacritics, strip leading emoji flag."""
    name = name.strip()
    if name and not name[0].isascii():
        parts = name.split(None, 1)
        name = parts[1] if len(parts) > 1 else ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()


_ALIAS: dict[str, str] = {
    "united states":        "usa",
    "united states of america": "usa",
    "ir iran":              "iran",
    "korea republic":       "south korea",
    "republic of ireland":  "ireland",
    "ivory coast":          "ivory coast",
    "cote d'ivoire":        "ivory coast",
    "cote divoire":         "ivory coast",
    "democratic republic of congo": "congo dr",
    "dr congo":             "congo dr",
    "cabo verde":           "cape verde islands",
    "cape verde":           "cape verde islands",
}


def _canonical(name: str) -> str:
    n = _norm(name)
    return _ALIAS.get(n, n)


# ── Data loading ──────────────────────────────────────────────────────────────

_cache: Optional[dict] = None


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(_PRIORS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        # Build normalised lookup: canonical_name → entry
        _cache = {_canonical(k): v for k, v in raw.items() if not k.startswith("_")}
    except FileNotFoundError:
        print(f"[opta] opta_priors.json not found at {_PRIORS_PATH} — Opta layer disabled.")
        _cache = {}
    except json.JSONDecodeError as e:
        print(f"[opta] Failed to parse opta_priors.json: {e} — Opta layer disabled.")
        _cache = {}
    return _cache


def get_team_opta(team_name: str) -> Optional[dict]:
    """
    Return the Opta probability dict for a team, or None if unknown / all-zero.

    Dict keys: win, final, semi, qf, r16 (all floats, % of simulations).
    """
    data = _load()
    entry = data.get(_canonical(team_name))
    if entry is None:
        return None
    # Treat an all-zero entry as "no data" (unfilled placeholder)
    if all(v == 0.0 for v in entry.values()):
        return None
    return entry


# ── Context injection ─────────────────────────────────────────────────────────

def build_opta_context(home_team: str, away_team: str) -> str:
    """
    Build a short context string for injection into the Claude prompt.
    Returns empty string if neither team has Opta data.
    """
    h = get_team_opta(home_team)
    a = get_team_opta(away_team)

    if h is None and a is None:
        return ""

    lines = ["🔭 OPTA SUPERCOMPUTER (25k simulations, tournament-wide):"]

    def _fmt(name: str, e: Optional[dict]) -> str:
        if e is None:
            return f"  {name}: No Opta data available"
        parts = []
        if e.get("win"):
            parts.append(f"win tournament {e['win']:.1f}%")
        if e.get("final"):
            parts.append(f"reach final {e['final']:.1f}%")
        if e.get("semi"):
            parts.append(f"reach semi {e['semi']:.1f}%")
        if e.get("qf"):
            parts.append(f"reach QF {e['qf']:.1f}%")
        if e.get("r16"):
            parts.append(f"reach R16 {e['r16']:.1f}%")
        return f"  {name}: " + (" | ".join(parts) if parts else "data partially available")

    lines.append(_fmt(home_team, h))
    lines.append(_fmt(away_team, a))
    lines.append(
        "  These are tournament survival probabilities, not match-specific win odds. "
        "Use them to calibrate relative team strength and flag if Poisson diverges significantly."
    )
    return "\n".join(lines)


# ── Sentiment gap detection ───────────────────────────────────────────────────

def detect_sentiment_gap(
    home_team: str,
    away_team: str,
    poisson_p_home: float,
    poisson_p_away: float,
) -> Optional[str]:
    """
    Compare our Poisson-implied match strength ratio to Opta's tournament strength ratio.
    Returns a formatted gap note for WhatsApp/Claude, or None if no gap detected.

    Method:
      - Opta strength ratio = home_win% / (home_win% + away_win%)  (normalised to 2-way)
      - Poisson strength ratio = p_home / (p_home + p_away)          (normalised to 2-way)
      - If |opta_ratio - poisson_ratio| > SENTIMENT_GAP_THRESHOLD → flag it
    """
    h = get_team_opta(home_team)
    a = get_team_opta(away_team)

    if h is None or a is None:
        return None
    if (h.get("win", 0) + a.get("win", 0)) == 0:
        return None

    opta_h_win = h.get("win", 0)
    opta_a_win = a.get("win", 0)
    opta_total = opta_h_win + opta_a_win
    if opta_total == 0:
        return None

    opta_ratio  = opta_h_win / opta_total        # Opta-implied home strength share
    poisson_total = poisson_p_home + poisson_p_away
    if poisson_total == 0:
        return None
    poisson_ratio = poisson_p_home / poisson_total  # Poisson-implied home strength share

    gap = poisson_ratio - opta_ratio              # +ve = our model favours home more than Opta

    if abs(gap) < SENTIMENT_GAP_THRESHOLD:
        return None

    direction = home_team if gap > 0 else away_team
    opta_favoured = away_team if gap > 0 else home_team
    gap_pct = abs(gap) * 100

    # Surface R16% for the Opta-favoured team as a knockout-resilience signal
    opta_fav_entry = a if opta_favoured == away_team else h
    r16_note = ""
    if opta_fav_entry and opta_fav_entry.get("r16"):
        r16_note = f" Opta R16 survival: {opta_fav_entry['r16']:.1f}%."

    return (
        f"⚠️ Market Sentiment Gap ({gap_pct:.0f}pp): Our Poisson model favours "
        f"*{direction}* more than the Opta supercomputer. "
        f"Opta implies *{opta_favoured}* is the stronger team on a tournament-wide basis "
        f"({opta_favoured} {opta_a_win if opta_favoured == away_team else opta_h_win:.1f}% "
        f"vs {direction} {opta_h_win if direction == home_team else opta_a_win:.1f}% to win tournament)."
        f"{r16_note} Consider whether lineup, context, or stage explains this deviation."
    )


# ── Statistical tiebreaker ────────────────────────────────────────────────────

def opta_tiebreak(
    home_team: str,
    away_team: str,
    poisson_p_home: float,
    poisson_p_draw: float,
    poisson_p_away: float,
) -> Optional[tuple[float, float, float]]:
    """
    When the simulation is indecisive (|p_home - p_away| < TIEBREAKER_THRESHOLD),
    nudge the probabilities using Opta's tournament strength ratio.

    Returns adjusted (p_home, p_draw, p_away) — or None if Opta data unavailable
    or simulation is already decisive.
    """
    if abs(poisson_p_home - poisson_p_away) >= TIEBREAKER_THRESHOLD:
        return None   # simulation is already decisive — don't touch it

    h = get_team_opta(home_team)
    a = get_team_opta(away_team)
    if h is None or a is None:
        return None

    opta_h = h.get("win", 0)
    opta_a = a.get("win", 0)
    if (opta_h + opta_a) == 0:
        return None

    # Compute Opta strength ratio and nudge home/away by up to 3pp (draw stays constant)
    opta_ratio = opta_h / (opta_h + opta_a)   # 0.0–1.0; > 0.5 means home stronger
    nudge      = (opta_ratio - 0.5) * 0.06    # max ±3pp nudge at extreme ratios

    adj_home = max(0.01, poisson_p_home + nudge)
    adj_away = max(0.01, poisson_p_away - nudge)
    # Renormalise so sum stays 1.0
    total = adj_home + poisson_p_draw + adj_away
    return adj_home / total, poisson_p_draw / total, adj_away / total


# ── WhatsApp-ready sentiment note ─────────────────────────────────────────────

def get_whatsapp_sentiment_note(
    home_team: str,
    away_team: str,
    poisson_p_home: float,
    poisson_p_away: float,
) -> Optional[str]:
    """Convenience wrapper — returns a short WhatsApp-safe sentiment note or None."""
    return detect_sentiment_gap(home_team, away_team, poisson_p_home, poisson_p_away)
