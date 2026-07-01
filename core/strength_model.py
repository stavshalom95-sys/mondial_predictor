"""
strength_model.py — Dixon-Coles simplified team strength ratings for the Poisson engine.

Formula (neutral-venue variant, appropriate for World Cup):

    λ_home = (H_attack × A_defence) / tournament_average
    λ_away = (A_attack × H_defence) / tournament_average

Where:
    H_attack  = home team's Bayesian-smoothed goals scored per game  (v2)
    A_defence = away team's raw goals conceded per game
    tournament_average = total goals / (2 × total matches)

Blending with market-calibrated λ (see main.py) prevents the model from
over-reacting to small samples early in the tournament:
    λ_blended = (1 - w) × λ_market  +  w × λ_strength

Bayesian Prior (v2):
    Attack rates are smoothed toward FIFA-ranking-derived priors to prevent
    extreme λ estimates from a single match result.

    posterior = (n_games × observed + PRIOR_WEIGHT × prior) / (n_games + PRIOR_WEIGHT)

    With PRIOR_WEIGHT=3, Norway's 4-goal game against Iraq:
        posterior_attack = (1×4.0 + 3×1.55) / 4 = 2.16
    vs the naive raw estimate of 4.0 goals/game.  The strength model then
    quality-adjusts this through the division by tournament_average.

Self-learning priors (v3):
    After each pipeline run, save_wc_priors() computes the current Bayesian
    posterior for every team and writes it to data/wc_priors.json.
    The next run loads this file and uses it as the prior seed — meaning the
    model compounds tournament-learning day over day rather than resetting to
    static FIFA estimates.

    This also serves as the bridge to European leagues: after the WC, the
    saved posteriors become the "national team pedigree" starting point for
    club-football priors.

Usage:
    from core.strength_model import build_strength_model, save_wc_priors, load_wc_priors
    wc_priors = load_wc_priors("data/wc_priors.json")
    model = build_strength_model(completed_matches, external_priors=wc_priors)
    if model:
        lam_h, lam_a = model.lambdas("Norway", "Senegal")
    save_wc_priors(completed_matches, "data/wc_priors.json")
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

MIN_MATCHES  = 3     # fewer than this → return None (not enough data)
MIN_BLEND    = 8     # fewer than this → skip blending in main.py, log info only
BLEND_WEIGHT = 0.20  # static fallback — prefer dynamic_blend_weight() in main.py


def dynamic_blend_weight(n_matches: int) -> float:
    """
    Returns the strength-model blend weight as a function of WC matches played.

    AI Research Skills: brainstorming-research-ideas Framework 5 (What Changed).
    Early tournament: few samples → trust market odds more (low weight).
    Late tournament:  many samples → historical strength is more informative.

    Growth: 5% at 0 matches → 35% at 60+ matches (linear ramp, capped).
    Formula: w = min(0.35, 0.05 + n_matches * 0.005)

    Examples:
        0  matches played → 5%   (almost pure market odds)
        8  matches played → 9%   (just past MIN_BLEND gate)
        20 matches played → 15%
        40 matches played → 25%
        60 matches played → 35%  (cap)
    """
    return round(min(0.35, 0.05 + n_matches * 0.005), 3)


# ---------------------------------------------------------------------------
# Bayesian prior: FIFA-ranking-derived baseline attack rate per team
# ---------------------------------------------------------------------------
# Calibrated to the long-run WC average of 1.32 goals/team/game (2.64/game).
# Keys are _norm()-ised team names (lowercase, emoji-flag stripped).
# Source: FIFA world rankings mapped to historical WC scoring rates.
# Priors updated 2026-06-28 after group stage observations.
#
# IMPORTANT: Multiple keys per team handle API name variants.
#   e.g. football-data.org sends "Turkey" but we also keep "türkiye".
#   Always add the _norm() of every known API variant as an alias.

_FIFA_PRIOR: dict[str, float] = {
    # ── Elite: FIFA top 10  (>1.50 g/game) ───────────────────────────────
    "france":                   1.72,   # obs: 3.5 g/game (4 vs Norway; dominant)
    "argentina":                1.60,
    "brazil":                   1.60,
    "england":                  1.48,   # obs: 0-0 draws, struggling to score
    "germany":                  1.44,   # obs: lost to Ecuador, below expectations
    "spain":                    1.55,
    "norway":                   1.48,   # obs: conceded 4 to France
    "netherlands":              1.50,
    "portugal":                 1.50,
    "belgium":                  1.45,

    # ── Strong: FIFA 11–30  (1.25–1.44 g/game) ───────────────────────────
    "croatia":                  1.40,
    "uruguay":                  1.35,
    "colombia":                 1.35,
    "usa":                      1.30,   # obs: lost 2-3 to Turkey
    "united states":            1.30,   # API variant of "usa"
    "mexico":                   1.35,
    "japan":                    1.32,   # obs: drew 1-1 vs Sweden
    "canada":                   1.30,
    "korea republic":           1.30,
    "south korea":              1.30,   # API variant of "korea republic"
    "switzerland":              1.30,
    "austria":                  1.25,
    "sweden":                   1.28,   # obs: held Japan 1-1
    "morocco":                  1.25,

    # ── Mid-tier: FIFA 31–55  (1.05–1.24 g/game) ─────────────────────────
    "ecuador":                  1.26,   # obs: beat Germany 2-1
    "scotland":                 1.15,
    "senegal":                  1.28,   # obs: 5-0 vs Iraq, 7 goals in 2 games
    "egypt":                    1.15,
    "côte d'ivoire":            1.15,
    "ivory coast":              1.15,   # API variant of "côte d'ivoire"
    "australia":                1.15,
    "ir iran":                  1.10,
    "iran":                     1.10,   # API variant of "ir iran"
    "türkiye":                  1.28,   # obs: beat USA 3-2, massively underrated
    "turkey":                   1.28,   # API variant of "türkiye"
    "czechia":                  1.10,
    "ghana":                    1.05,
    "south africa":             1.12,   # obs: beat South Korea
    "algeria":                  1.05,
    "congo dr":                 1.05,

    # ── Lower-mid: FIFA 56–90  (0.90–1.04 g/game) ────────────────────────
    "paraguay":                 1.00,
    "jordan":                   1.00,
    "bosnia and herzegovina":   1.00,
    "bosnia-herzegovina":       1.00,   # API variant (hyphen)
    "new zealand":              0.95,
    "saudi arabia":             0.95,
    "tunisia":                  0.95,
    "panama":                   0.92,
    "cabo verde":               0.90,
    "cape verde islands":       0.90,   # API variant of "cabo verde"
    "uzbekistan":               0.90,
    "qatar":                    0.90,

    # ── Weak: FIFA 91+  (<0.90 g/game) ───────────────────────────────────
    "iraq":                     0.70,   # obs: 0 goals in 2 games, 8 conceded
    "curaçao":                  0.85,
    "haiti":                    0.85,
}

_DEFAULT_PRIOR: float = 1.05   # fallback for unlisted teams
_PRIOR_WEIGHT:  float = 3.0    # equivalent to 3 "virtual" prior games

# Hard caps on posterior values saved by save_wc_priors().
# Prevents outlier matches from inflating a team's prior permanently.
# Example: Senegal 5-0 vs Iraq (red card, 10 men) → inflated posterior ≈1.97
# Cap: Senegal's saved prior cannot exceed 1.40.
_POSTERIOR_CAPS: dict[str, float] = {
    "senegal": 1.40,   # outlier: 5-0 vs Iraq (10 men, red card in 1st half)
}


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


def _bayesian_attack(
    observed_avg: float,
    team_key: str,
    n_games: int,
    priors: dict[str, float] | None = None,
) -> float:
    """
    Bayesian posterior for a team's attack rate.

    Smooths the observed goals/game toward a prior to prevent extreme λ
    estimates from small match samples.

    Formula:
        posterior = (n_games × observed + PRIOR_WEIGHT × prior) / (n_games + PRIOR_WEIGHT)

    Args:
        observed_avg: Raw goals/game from WC matches played so far.
        team_key:     Normalised team name (output of _norm()).
        n_games:      Number of WC games this team has played.
        priors:       Optional dict of {team_key: prior_rate}.  If provided,
                      takes precedence over _FIFA_PRIOR (external_priors flow).

    Returns:
        Bayesian-smoothed attack rate (goals/game).
    """
    if priors is not None:
        prior = priors.get(team_key, _FIFA_PRIOR.get(team_key, _DEFAULT_PRIOR))
    else:
        prior = _FIFA_PRIOR.get(team_key, _DEFAULT_PRIOR)
    return (n_games * observed_avg + _PRIOR_WEIGHT * prior) / (n_games + _PRIOR_WEIGHT)


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
    Holds per-team WC stats and computes expected goals for any fixture.

    Attributes
    ----------
    n_matches   : number of completed matches used to build the model
    avg_goals   : tournament average goals per team per game (= λ baseline)
    _priors     : merged prior dict (external_priors ∪ _FIFA_PRIOR).
                  Stored here so lambdas() uses the same prior that was used
                  at construction time.
    """
    _stats:    dict[str, _TeamStats] = field(repr=False)
    avg_goals: float
    n_matches: int
    _priors:   dict[str, float]      = field(default_factory=dict, repr=False)

    def lambdas(self, home_team: str, away_team: str) -> tuple[float, float]:
        """
        Return (λ_home, λ_away) using Bayesian-smoothed attack rates.

        Attack rates use _bayesian_attack() to prevent over-reaction to
        single-game samples.  Defence uses Bayesian smoothing toward the
        tournament average (prevents 0-conceded teams from driving λ to floor).

        Falls back to avg_goals for teams with no WC matches recorded.
        """
        avg   = self.avg_goals or 1.3
        h_key = _norm(home_team)
        a_key = _norm(away_team)
        h     = self._stats.get(h_key, _TeamStats())
        a     = self._stats.get(a_key, _TeamStats())

        # Bayesian-smoothed attack rate (uses self._priors as the prior seed)
        h_atk = _bayesian_attack(h.attack if h.games else avg, h_key, h.games, self._priors or None)
        a_atk = _bayesian_attack(a.attack if a.games else avg, a_key, a.games, self._priors or None)

        # Bayesian-smoothed defence rate toward tournament average.
        # Raw rate collapses to 0.0 when a team hasn't conceded yet (common
        # early in tournament), driving λ to the 0.10 floor and producing
        # nonsense ~82% draw probabilities.  Smoothing prevents this.
        h_def = (h.games * h.defence + _PRIOR_WEIGHT * avg) / (h.games + _PRIOR_WEIGHT) if h.games else avg
        a_def = (a.games * a.defence + _PRIOR_WEIGHT * avg) / (a.games + _PRIOR_WEIGHT) if a.games else avg

        # Dixon-Coles neutral-venue formula
        # λ = (scorer_attack × conceder_defence) / tournament_average
        lam_h = h_atk * a_def / avg
        lam_a = a_atk * h_def / avg

        return round(max(lam_h, 0.10), 3), round(max(lam_a, 0.10), 3)

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
    external_priors: dict[str, float] | None = None,
) -> StrengthModel | None:
    """
    Build a StrengthModel from a list of completed match dicts.

    Parameters
    ----------
    matches         : list of dicts with home_team, away_team, and goal counts.
                      Accepts both 'home_goals'/'away_goals' and 'actual_home'/'actual_away'.
    last_n          : if > 0, use only the most recent n matches (rolling window).
    external_priors : optional dict {normalised_team_name: attack_rate}.
                      Takes precedence over _FIFA_PRIOR for listed teams.
                      Used for self-learning: pass load_wc_priors() output here
                      so the model compounds daily tournament learning.

    Returns None when fewer than MIN_MATCHES valid results are found.
    """
    valid = [m for m in matches if _extract(m) != (None, None)]

    if last_n > 0:
        valid = valid[-last_n:]

    if len(valid) < MIN_MATCHES:
        return None

    # Merge external_priors over _FIFA_PRIOR (external wins)
    merged_priors: dict[str, float] = {**_FIFA_PRIOR, **(external_priors or {})}

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
    avg = total_goals / (2 * n) if n else 1.3
    avg = max(avg, 0.3)

    return StrengthModel(
        _stats    = stats,
        avg_goals = round(avg, 3),
        n_matches = n,
        _priors   = merged_priors,
    )


# ---------------------------------------------------------------------------
# Self-learning: persist and reload WC-informed priors
# ---------------------------------------------------------------------------

def save_wc_priors(all_matches: list[dict], path: str) -> None:
    """
    Compute each team's current Bayesian posterior attack rate from all WC
    matches played so far, and save to a JSON file.

    Called at the END of each pipeline run so the next run starts with
    tournament-informed priors instead of static FIFA estimates.

    The saved values are posteriors — they will be used as the PRIOR for the
    next run, giving the model compounding day-over-day learning:

        Day 1 prior  = _FIFA_PRIOR
        Day 1 result → save posterior_day1 → wc_priors.json
        Day 2 prior  = wc_priors.json (= posterior_day1)
        Day 2 result → save posterior_day2 → wc_priors.json
        ...

    For leagues: after WC ends, wc_priors.json represents each national team's
    demonstrated scoring rate — a ready-made prior for club football if
    adapted by club→nation mapping.
    """
    model = build_strength_model(all_matches)
    if model is None:
        return

    priors: dict[str, float] = {}
    for team_key, s in model._stats.items():
        if s.games > 0:
            posterior = _bayesian_attack(s.attack, team_key, s.games, model._priors)
            posterior = min(posterior, _POSTERIOR_CAPS.get(team_key, float("inf")))
            priors[team_key] = round(posterior, 4)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(priors, f, indent=2, ensure_ascii=False)

    print(f"[strength] wc_priors.json saved — {len(priors)} teams, path={path}")


def load_wc_priors(path: str) -> dict[str, float]:
    """
    Load previously saved WC priors from JSON.

    Returns an empty dict (= fall back to _FIFA_PRIOR) if the file doesn't
    exist (first run of the tournament) or is unreadable.

    Usage in main.py:
        wc_priors     = load_wc_priors("data/wc_priors.json")
        strength_model = build_strength_model(results, external_priors=wc_priors)
        # ... end of pipeline ...
        save_wc_priors(results, "data/wc_priors.json")
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items()}
    except Exception as exc:
        print(f"[strength] Warning: could not load wc_priors.json: {exc}")
        return {}
