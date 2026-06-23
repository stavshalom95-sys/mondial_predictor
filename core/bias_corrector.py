"""
core/bias_corrector.py — Per-team systematic goal-prediction bias corrector.

Reads data/history.json records and computes per-team average prediction
error (predicted_goals_for − actual_goals_for). Applies a conservative
correction to λ to counteract systematic over/under-prediction.

Correction is only applied when:
  • ≥ BIAS_MIN_MATCHES finished WC records exist for the team
  • |avg_error| ≥ BIAS_THRESHOLD (avoids noise corrections)

The offset is added to λ before the Poisson matrix is rebuilt:
  lam_adjusted = max(0.1, lam + offset)

  offset > 0  → model historically under-predicted goals → nudge λ up
  offset < 0  → model historically over-predicted goals  → nudge λ down

Public API:
    build_bias_corrector(history: list[dict]) -> BiasCorrector
    BiasCorrector.get_offset(team_name: str)  -> float
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

# ── Constants ──────────────────────────────────────────────────────────────────

BIAS_MIN_MATCHES: int   = 2    # require ≥ 2 WC records before trusting the average
BIAS_THRESHOLD:   float = 0.3  # min |avg_error| to apply any correction
BIAS_DECAY:       float = 0.5  # apply 50% of observed error as the λ correction


# ── Helpers ────────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Strip emoji/flags, diacritics, lowercase — mirrors performance_tracker._normalize."""
    filtered = "".join(
        c for c in name
        if unicodedata.category(c).startswith(("L", "N", "Z", "P"))
    )
    nfd      = unicodedata.normalize("NFD", " ".join(filtered.split()))
    no_marks = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return " ".join(no_marks.lower().split())


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class BiasCorrector:
    """Holds per-team λ offsets computed from WC prediction history."""

    _offsets: dict[str, float] = field(default_factory=dict)

    def get_offset(self, team_name: str) -> float:
        """
        Return the λ correction offset for team_name.
        Returns 0.0 when the team has insufficient data or negligible error.
        """
        return self._offsets.get(_norm(team_name), 0.0)

    def __len__(self) -> int:
        return len(self._offsets)

    def summary(self) -> str:
        if not self._offsets:
            return "[bias] No bias corrections active (insufficient data or errors within threshold)."
        lines = ["[bias] Per-team λ corrections:"]
        for team, off in sorted(self._offsets.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "↑ under-predicted" if off > 0 else "↓ over-predicted"
            lines.append(f"  {team}: {off:+.3f}  ({direction})")
        return "\n".join(lines)


# ── Core logic ─────────────────────────────────────────────────────────────────

def build_bias_corrector(history: list[dict]) -> BiasCorrector:
    """
    Compute per-team goal-prediction bias from history.json records.

    For every finished WC match in history, records the per-team goal error:
      home error = predicted_home − actual_home
      away error = predicted_away − actual_away

    Teams with ≥ BIAS_MIN_MATCHES records and |avg_error| ≥ BIAS_THRESHOLD
    receive a λ offset of −avg_error × BIAS_DECAY applied before the Poisson
    matrix is rebuilt in main.py.

    Args:
        history:  List of records from data/history.json.

    Returns:
        BiasCorrector with per-team offsets (empty if no data or below threshold).
    """
    # errors[norm_team] = list of per-game (predicted_goals_for − actual_goals_for)
    errors: dict[str, list[float]] = {}

    for rec in history:
        home = rec.get("home_team", "")
        away = rec.get("away_team", "")
        ph   = rec.get("predicted_home")
        pa   = rec.get("predicted_away")
        ah   = rec.get("actual_home")
        aa   = rec.get("actual_away")

        if None in (home, away, ph, pa, ah, aa):
            continue
        try:
            ph, pa, ah, aa = int(ph), int(pa), int(ah), int(aa)
        except (TypeError, ValueError):
            continue

        errors.setdefault(_norm(home), []).append(float(ph - ah))  # home: predicted − actual
        errors.setdefault(_norm(away), []).append(float(pa - aa))  # away: predicted − actual

    offsets: dict[str, float] = {}
    for team, errs in errors.items():
        if len(errs) < BIAS_MIN_MATCHES:
            continue
        avg_err = sum(errs) / len(errs)
        if abs(avg_err) < BIAS_THRESHOLD:
            continue
        # offset = negative error × decay:
        #   over-predicted (avg_err > 0)  → offset < 0 → λ pushed down
        #   under-predicted (avg_err < 0) → offset > 0 → λ pushed up
        offsets[team] = round(-avg_err * BIAS_DECAY, 3)

    corrector = BiasCorrector(_offsets=offsets)
    print(corrector.summary())
    return corrector
