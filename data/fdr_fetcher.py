"""
fdr_fetcher.py — Fetch per-fixture expected goals from vice-captain.com.

Endpoint: GET https://vice-captain.com/api/vc/wc/goals-cs
Returns 80 WC 2026 fixtures with muHome / muAway (FDR-derived expected goals).

These mu values are used as a Strength Modifier: after the Poisson model is
calibrated from bookmaker odds, we blend the calibrated lambdas with the
FDR signal via apply_fdr_modifier().  Alpha=0.15 means "15% weight to FDR".

Example:
  Odds-calibrated:  lambda_home=2.10, lambda_away=0.65
  FDR signal:       mu_home=3.22,     mu_away=0.47   (Spain vs Saudi Arabia)
  Blended (α=0.15): lambda_home=2.27, lambda_away=0.62

The fixture list is fetched once and cached in-process for the lifetime of the
pipeline run (one morning run ≈ one process), so at most 1 HTTP call per run.
"""
from __future__ import annotations

import unicodedata
from typing import Optional

import requests

from core.poisson_engine import PoissonMatchModel, _build_matrix

_FDR_URL  = "https://vice-captain.com/api/vc/wc/goals-cs"
_TIMEOUT  = 10

# ---------------------------------------------------------------------------
# Team-name aliases  (API name → canonical lower-case key)
# The FDR API uses its own spellings; this maps them to normalised forms that
# also cover football-data.org / The Odds API / schedule team names.
# ---------------------------------------------------------------------------
_ALIASES: dict[str, str] = {
    # USA
    "usa":                          "usa",
    "united states":                "usa",
    "us":                           "usa",
    # Turkey
    "turkey":                       "turkiye",
    "türkiye":                      "turkiye",
    "turkiye":                      "turkiye",
    # Curaçao
    "curacao":                      "curacao",
    "curaçao":                      "curacao",
    # Ivory Coast
    "ivory coast":                  "ivory coast",
    "côte d'ivoire":                "ivory coast",
    "cote d'ivoire":                "ivory coast",
    "cote divoire":                 "ivory coast",
    # DR Congo
    "dr congo":                     "dr congo",
    "democratic republic of congo": "dr congo",
    "dr. congo":                    "dr congo",
    "congo dr":                     "dr congo",
    # Cape Verde
    "cape verde":                   "cape verde",
    "cape verde islands":           "cape verde",
    "cabo verde":                   "cape verde",
    # South Korea
    "south korea":                  "south korea",
    "korea republic":               "south korea",
    "republic of korea":            "south korea",
    # Bosnia
    "bosnia and herzegovina":       "bosnia",
    "bosnia & herzegovina":         "bosnia",
    "bosnia":                       "bosnia",
    # Iran
    "iran":                         "iran",
    "ir iran":                      "iran",
    # North Macedonia
    "north macedonia":              "north macedonia",
    "macedonia":                    "north macedonia",
    # Others that appear in odds APIs
    "czechia":                      "czechia",
    "czech republic":               "czechia",
}


# ---------------------------------------------------------------------------
# In-process cache (populated on first call, reused for all matches in the run)
# ---------------------------------------------------------------------------
_fixture_cache: Optional[list[dict]] = None


def _load_fixtures() -> list[dict]:
    """Fetch (or return cached) fixture list from the FDR API."""
    global _fixture_cache
    if _fixture_cache is not None:
        return _fixture_cache

    try:
        resp = requests.get(_FDR_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[fdr] Request failed: {exc}")
        _fixture_cache = []
        return []

    if not data.get("success"):
        print(f"[fdr] API returned success=false")
        _fixture_cache = []
        return []

    _fixture_cache = data.get("fixtures", [])
    valid = sum(
        1 for f in _fixture_cache
        if f.get("muHome") is not None
    )
    print(f"[fdr] Loaded {len(_fixture_cache)} fixtures ({valid} with mu values).")
    return _fixture_cache


def _normalize(name: str) -> str:
    """Lowercase, strip diacritics, apply alias table."""
    nfd = unicodedata.normalize("NFD", name)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    key = " ".join(stripped.lower().split())
    return _ALIASES.get(key, key)


def _teams_match(schedule_name: str, api_name: str) -> bool:
    """True if two team strings refer to the same team after normalisation."""
    ns = _normalize(schedule_name)
    na = _normalize(api_name)
    return ns == na or ns in na or na in ns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_fixture_mu(
    home_team: str,
    away_team: str,
) -> Optional[tuple[float, float]]:
    """
    Return (mu_home, mu_away) expected goals for this fixture from vice-captain.com.

    Tries exact-order match first, then swapped (neutral-venue edge case).
    Returns None if:
      - the fixture is not found in the API response, or
      - the fixture has null mu values (already played / no market), or
      - any network / parse error occurred.
    """
    fixtures = _load_fixtures()

    # 1. Try normal order (home = schedule home)
    for fx in fixtures:
        if (
            fx.get("muHome") is not None
            and fx.get("muAway") is not None
            and _teams_match(home_team,  fx.get("homeName", ""))
            and _teams_match(away_team,  fx.get("awayName", ""))
        ):
            mu_h = float(fx["muHome"])
            mu_a = float(fx["muAway"])
            print(f"[fdr]   {home_team} vs {away_team}: mu_home={mu_h:.2f}, mu_away={mu_a:.2f}")
            return mu_h, mu_a

    # 2. Try swapped order
    for fx in fixtures:
        if (
            fx.get("muHome") is not None
            and fx.get("muAway") is not None
            and _teams_match(home_team,  fx.get("awayName", ""))
            and _teams_match(away_team,  fx.get("homeName", ""))
        ):
            # Swap back so mu_home corresponds to our schedule's home team
            mu_h = float(fx["muAway"])
            mu_a = float(fx["muHome"])
            print(
                f"[fdr]   {home_team} vs {away_team}: mu_home={mu_h:.2f}, mu_away={mu_a:.2f} "
                f"(API order was swapped)"
            )
            return mu_h, mu_a

    print(f"[fdr]   No FDR mu found for '{home_team} vs {away_team}' — modifier skipped.")
    return None


def apply_fdr_modifier(
    model: PoissonMatchModel,
    mu_home: float,
    mu_away: float,
    alpha: float = 0.15,
) -> PoissonMatchModel:
    """
    Blend Poisson lambdas calibrated from bookmaker odds with FDR expected goals.

    alpha = weight given to the FDR signal (0 = ignore FDR, 1 = use FDR only).
    Default 0.15 → "slightly increase/decrease expected goals" as per project spec.

    A team with a high FDR mu (easy fixture → many goals expected) will have its
    lambda nudged upward; a team facing a strong opponent will have theirs nudged down.

    Returns a new PoissonMatchModel — the original is never mutated.
    """
    lh = model.lambda_home * (1.0 - alpha) + mu_home * alpha
    la = model.lambda_away * (1.0 - alpha) + mu_away * alpha
    lh = max(lh, 0.05)  # safety floor
    la = max(la, 0.05)
    matrix = _build_matrix(lh, la)
    print(
        f"[fdr]   lambda_home: {model.lambda_home:.2f} → {lh:.2f}  "
        f"lambda_away: {model.lambda_away:.2f} → {la:.2f}  (α={alpha})"
    )
    return PoissonMatchModel(lambda_home=lh, lambda_away=la, _matrix=matrix)
