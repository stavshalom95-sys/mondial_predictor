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
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────────

BIAS_MIN_MATCHES: int   = 3    # require ≥ 3 WC records before trusting the average
BIAS_THRESHOLD:   float = 0.25 # min |avg_error| to apply any correction (was 0.30)
BIAS_DECAY:       float = 0.55 # apply 55% of observed error as the λ correction (was 0.50)

# ── Goal-rate scaler constants ─────────────────────────────────────────────────
# SCALE_BLEND is deliberately low (0.30) because per-team bias already accounts
# for most team-specific under/over-prediction. The global scaler is a gentle
# residual correction for teams with insufficient history (< BIAS_MIN_MATCHES).
SCALE_MIN_MATCHES: int   = 10    # min games before trusting the tournament-wide scale
SCALE_BLEND:       float = 0.30  # conservative blend: 30% of observed deviation applied
SCALE_MAX:         float = 2.0   # upper cap to prevent extreme values
SCALE_MIN_SCALE:   float = 0.50  # lower cap


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
        lines = ["[bias] Per-team lam corrections:"]
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


# ── Global goal-rate scaler ────────────────────────────────────────────────────

@dataclass
class GoalRateScaler:
    """
    Tournament-wide multiplicative λ correction derived from actual vs predicted
    goals-per-game across all history records.

    If the model has been systematically under-predicting goals (actual_avg >
    predicted_avg), scale > 1.0 and all λ values are nudged up.  The blend is
    conservative: only SCALE_BLEND of the observed deviation is applied.

    Example (WC 2026 after 44 games):
        predicted_avg = 2.0  actual_avg = 3.2  → raw_scale = 1.60
        blended_scale = 1.0 + 0.65 × (1.60 − 1.0) = 1.39
    """
    scale:         float
    n_samples:     int
    predicted_avg: float
    actual_avg:    float

    def apply(self, lam: float) -> float:
        """Scale a single λ value; identity when scale ≈ 1.0."""
        if abs(self.scale - 1.0) < 0.005:
            return lam
        return round(max(0.10, lam * self.scale), 3)

    def summary(self) -> str:
        if self.n_samples < SCALE_MIN_MATCHES:
            return (
                f"[goal_scale] Identity — only {self.n_samples} sample(s) "
                f"(need ≥{SCALE_MIN_MATCHES})."
            )
        if abs(self.scale - 1.0) < 0.005:
            return "[goal_scale] No adjustment — predicted/actual within tolerance."
        direction = (
            "under-predicting goals → λ scaled UP"
            if self.scale > 1.0
            else "over-predicting goals → λ scaled DOWN"
        )
        return (
            f"[goal_scale] scale={self.scale:.3f}  "
            f"({self.predicted_avg:.2f} predicted → {self.actual_avg:.2f} actual "
            f"goals/game, n={self.n_samples})  — {direction}"
        )


def build_goal_rate_scaler(history: list[dict]) -> GoalRateScaler:
    """
    Compute a tournament-wide λ scaling factor from prediction history.

    For each finished match record, computes predicted total goals and actual
    total goals. Derives raw scale = actual_avg / predicted_avg, then blends
    conservatively with 1.0 to avoid over-fitting on early data.

    Args:
        history: list of records from data/history.json.

    Returns:
        GoalRateScaler (scale=1.0 when insufficient data).
    """
    pairs: list[tuple[int, int]] = []  # (predicted_total, actual_total)
    for rec in history:
        try:
            ph = int(rec["predicted_home"])
            pa = int(rec["predicted_away"])
            ah = int(rec["actual_home"])
            aa = int(rec["actual_away"])
            pairs.append((ph + pa, ah + aa))
        except (KeyError, TypeError, ValueError):
            continue

    n = len(pairs)
    if n < SCALE_MIN_MATCHES:
        scaler = GoalRateScaler(scale=1.0, n_samples=n, predicted_avg=0.0, actual_avg=0.0)
        print(scaler.summary())
        return scaler

    pred_avg   = sum(p for p, _ in pairs) / n
    actual_avg = sum(a for _, a in pairs) / n

    if pred_avg < 0.1:
        scaler = GoalRateScaler(scale=1.0, n_samples=n, predicted_avg=pred_avg, actual_avg=actual_avg)
        print(scaler.summary())
        return scaler

    raw_scale = actual_avg / pred_avg
    blended   = 1.0 + SCALE_BLEND * (raw_scale - 1.0)
    blended   = max(SCALE_MIN_SCALE, min(SCALE_MAX, blended))

    scaler = GoalRateScaler(
        scale         = round(blended, 3),
        n_samples     = n,
        predicted_avg = round(pred_avg, 3),
        actual_avg    = round(actual_avg, 3),
    )
    print(scaler.summary())
    return scaler
