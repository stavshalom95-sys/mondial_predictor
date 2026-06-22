"""
strength_model.py — Dixon-Coles simplified team strength ratings for the Poisson engine.

Formula (neutral-venue variant, appropriate for World Cup):

    λ_home = (H_attack × A_defence) / tournament_average
    λ_away = (A_attack × H_defence) / tournament_average

Where:
    H_attack  = home team's goals scored  per game
    A_defence = away team's goals conceded per game
    tournament_average = total goals / (2 × total matches)

Blending with market-calibrated λ (see main.py) prevents the model from
over-reacting to small samples early in the tournament:
    λ_blended = (1 - w) × λ_market  +  w × λ_strength

Usage:
    from core.strength_model import build_strength_model
    model = build_strength_model(completed_matches)   # list of dicts
    if model:
        lam_h, lam_a = model.lambdas("France", "Morocco")
"""
from __future__ import annotations

from dataclasses import dataclass, field

MIN_MATCHES = 3         # fewer than this → return None (not enough data)
MIN_BLEND   = 8         # fewer than this → skip blending, print info only
BLEND_WEIGHT = 0.20     # 80 % market odds  +  20 % historical strength


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Strip leading emoji-flag token and lowercase — identical to winner_odds_loader._norm."""
    name = name.strip()
    if name and not name[0].isascii():
        parts = name.split(None, 1)
        name = parts[1] if len(parts) > 1 else ""
    return name.lower().strip()


def _extract(match: dict) -> tuple[int | None, int | None]:
    """
    Return (home_goals, away_goals), accepting both:
      - schedule format  : keys home_goals / away_goals
      - history.json     : keys actual_home / actual_away
    """
    hg = match.get("home_goals", match.get("actual_home"))
    ag = match.get("away_goals", match.get("actual_away"))
    try:
        return int(hg), int(ag)
    except (TypeError, ValueError):
        return None, None


# ---------------------------------------------------------------------------
# Internal stats bucket (one per team)
# ---------------------------------------------------------------------------

@dataclass
class _TeamStats:
    scored:   int = 0
    conceded: int = 0
    games:    int = 0

    @property
    def attack(self) -> float:
        return self.scored   / self.games if self.games else 0.0

    @property
    def defence(self) -> float:
        return self.conceded / self.games if self.games else 0.0


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------

@dataclass
class StrengthModel:
    """
    Holds per-team stats and computes expected goals for any fixture.

    Attributes
    ----------
    n_matches   : number of completed matches used to build the model
    avg_goals   : tournament average goals per team per game (= λ baseline)
    """
    _stats:    dict[str, _TeamStats] = field(repr=False)
    avg_goals: float                  # tournament avg goals per team per game
    n_matches: int

    # ------------------------------------------------------------------
    def lambdas(self, home_team: str, away_team: str) -> tuple[float, float]:
        """
        Return (λ_home, λ_away) from strength ratings.

        Falls back to avg_goals for teams with no prior WC appearances.
        """
        avg = self.avg_goals or 1.3
        h   = self._stats.get(_norm(home_team), _TeamStats())
        a   = self._stats.get(_norm(away_team), _TeamStats())

        h_atk = h.attack  if h.games else avg
        h_def = h.defence if h.games else avg
        a_atk = a.attack  if a.games else avg
        a_def = a.defence if a.games else avg

        # λ = (scorer_attack × conceder_defence) / avg
        lam_h = h_atk * a_def / avg
        lam_a = a_atk * h_def / avg

        # safety floor: always ≥ 0.1 goals expected
        return round(max(lam_h, 0.10), 3), round(max(lam_a, 0.10), 3)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """One-line diagnostic string for pipeline logs."""
        top = sorted(
            ((nm, s) for nm, s in self._stats.items() if s.games),
            key=lambda kv: kv[1].attack,
            reverse=True,
        )[:3]
        top_str = "  ".join(f"{nm}(atk={s.attack:.2f})" for nm, s in top)
        return (
            f"[strength] {self.n_matches} WC matches · "
            f"avg={self.avg_goals:.2f} g/team/game · "
            f"top attack: {top_str}"
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_strength_model(
    matches: list[dict],
    last_n: int = 0,
) -> StrengthModel | None:
    """
    Build a StrengthModel from a list of completed match dicts.

    Parameters
    ----------
    matches : list of dicts containing home_team, away_team, and goal counts
              (accepts both 'home_goals'/'away_goals' and 'actual_home'/'actual_away')
    last_n  : if > 0, use only the most recent n matches (rolling window)

    Returns None when fewer than MIN_MATCHES valid results are found.
    """
    valid = [m for m in matches if _extract(m) != (None, None)]

    if last_n > 0:
        valid = valid[-last_n:]

    if len(valid) < MIN_MATCHES:
        return None

    stats:       dict[str, _TeamStats] = {}
    total_goals: int = 0

    for m in valid:
        ht = _norm(m.get("home_team", ""))
        at = _norm(m.get("away_team", ""))
        hg, ag = _extract(m)

        if not ht or not at or hg is None or ag is None:
            continue

        if ht not in stats:
            stats[ht] = _TeamStats()
        if at not in stats:
            stats[at] = _TeamStats()

        stats[ht].scored   += hg;  stats[ht].conceded += ag;  stats[ht].games += 1
        stats[at].scored   += ag;  stats[at].conceded += hg;  stats[at].games += 1
        total_goals        += hg + ag

    n   = len(valid)
    avg = total_goals / (2 * n) if n else 1.3  # goals per team per game
    avg = max(avg, 0.3)                          # safety floor

    return StrengthModel(_stats=stats, avg_goals=round(avg, 3), n_matches=n)
