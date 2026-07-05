"""
core/calibration.py — Temperature-scaling probability calibrator.

AI Research Skills applied:
  - brainstorming-research-ideas Framework 4 (Cross-Pollination from ML):
      Platt 1999 / Guo et al. 2017 "On Calibration of Modern Neural Networks"
      applies directly to Poisson-derived probabilities.
  - brainstorming-research-ideas Framework 6 (Failure Analysis):
      Poisson models are systematically overconfident on home-win probability
      in high-stakes matches; temperature scaling corrects this.

Theory (Temperature Scaling):
  Raw Poisson gives logit p = log(p / (1-p)).
  Calibrated:  p_cal = sigmoid( logit(p) / T )
  T > 1 → softer, less confident (model was overconfident)
  T < 1 → sharper, more confident (model was under-confident)
  T = 1 → identity (no correction)

Fitting:
  Minimise binary cross-entropy over historical (p_predicted, outcome) pairs,
  where outcome = 1 if the predicted 1X2 outcome actually occurred, 0 otherwise.
  Uses scipy.optimize.minimize_scalar (bracket search on T ∈ [0.5, 3.0]).

Fallback:
  Returns T=1.0 (identity) when:
    • scipy not available
    • fewer than MIN_SAMPLES records in history
    • optimisation fails

Public API:
    cal = build_calibrator(history)
    p_home_cal, p_draw_cal, p_away_cal = cal.calibrate(p_home, p_draw, p_away)
    print(cal.summary())
"""
from __future__ import annotations

import math
from dataclasses import dataclass

MIN_SAMPLES = 20    # require ≥ this many records before trusting the fitted T
_T_BOUNDS   = (0.30, 3.0)  # widened — optimizer was wall-pinned at lower bound
_EPS        = 1e-7   # clamp probabilities away from 0/1 to avoid log(0)


# ── Math helpers ─────────────────────────────────────────────────────────────

def _logit(p: float) -> float:
    p = max(_EPS, min(1.0 - _EPS, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _temp_scale(p: float, T: float) -> float:
    """Apply temperature T to a single probability."""
    if T == 1.0:
        return p
    return _sigmoid(_logit(p) / T)


def _renorm(ph: float, pd: float, pa: float) -> tuple[float, float, float]:
    """Normalise three probabilities to sum to 1.0."""
    total = ph + pd + pa
    if total <= 0:
        return 1/3, 1/3, 1/3
    return ph / total, pd / total, pa / total


# ── Public class ─────────────────────────────────────────────────────────────

@dataclass
class ProbabilityCalibrator:
    """
    Temperature-scaling calibrator fitted from WC prediction history.

    Attributes
    ----------
    temperature : float
        Fitted T.  1.0 = no adjustment.  >1 = model was overconfident.
    n_samples   : int
        Number of 1X2 outcome samples used to fit T.
    """
    temperature: float
    n_samples:   int

    def calibrate(
        self,
        p_home: float,
        p_draw: float,
        p_away: float,
    ) -> tuple[float, float, float]:
        """
        Apply temperature scaling and re-normalise so probabilities sum to 1.

        Args:
            p_home, p_draw, p_away: Raw Poisson 1X2 probabilities.

        Returns:
            Calibrated (p_home, p_draw, p_away) summing to 1.
        """
        T = self.temperature
        if abs(T - 1.0) < 1e-6:
            return p_home, p_draw, p_away
        ph = _temp_scale(p_home, T)
        pd = _temp_scale(p_draw, T)
        pa = _temp_scale(p_away, T)
        return _renorm(ph, pd, pa)

    def summary(self) -> str:
        if self.n_samples < MIN_SAMPLES:
            return (
                f"[calibration] Identity (T=1.0) — only {self.n_samples} samples "
                f"(need ≥{MIN_SAMPLES})."
            )
        direction = (
            "overconfident → softened" if self.temperature > 1.05
            else ("underconfident → sharpened" if self.temperature < 0.95
            else "well-calibrated")
        )
        return (
            f"[calibration] T={self.temperature:.3f} fitted on {self.n_samples} outcomes "
            f"({direction})"
        )


# ── Builder ───────────────────────────────────────────────────────────────────

def build_calibrator(history: list[dict]) -> ProbabilityCalibrator:
    """
    Fit temperature T from prediction history records.

    Each record must have:
        predicted_home  : int
        predicted_away  : int
        actual_home     : int
        actual_away     : int

    For every record we reconstruct the model's 1X2 outcome for the actual result
    and use it as a positive label (y=1). The raw Poisson probability at calibration
    time is not stored, so we approximate it from the predicted score position:
        predicted home win  → p_home proxy = 0.55
        predicted draw      → p_draw proxy = 0.30
        predicted away win  → p_away proxy = 0.55 (for away)
    These proxies are imprecise but still capture systematic over/underconfidence
    since the temperature correction is a global multiplier.

    For a higher-quality fit, callers should store the Poisson probabilities
    in morning_picks.json (fields: poisson_p_home, poisson_p_draw, poisson_p_away).

    Args:
        history : list of records from data/history.json.

    Returns:
        ProbabilityCalibrator with fitted T (or T=1.0 if insufficient data).
    """
    # Build (p_predicted_for_actual_outcome, 1) pairs from history
    samples: list[tuple[float, int]] = []

    for rec in history:
        try:
            ph = int(rec["predicted_home"])
            pa = int(rec["predicted_away"])
            ah = int(rec["actual_home"])
            aa = int(rec["actual_away"])
        except (KeyError, TypeError, ValueError):
            continue

        # Determine predicted and actual outcomes
        if ph > pa:
            pred_outcome = "home"
        elif ph == pa:
            pred_outcome = "draw"
        else:
            pred_outcome = "away"

        if ah > aa:
            actual_outcome = "home"
        elif ah == aa:
            actual_outcome = "draw"
        else:
            actual_outcome = "away"

        # Use stored Poisson probs if available; otherwise use proxies.
        # Records written by main.py use "sim_p_*" keys; legacy records used
        # "poisson_p_*" — try both so we use real data when present.
        p_home_raw = rec.get("sim_p_home") or rec.get("poisson_p_home")
        p_draw_raw = rec.get("sim_p_draw") or rec.get("poisson_p_draw")
        p_away_raw = rec.get("sim_p_away") or rec.get("poisson_p_away")

        if p_home_raw and p_draw_raw and p_away_raw:
            prob_map = {
                "home": float(p_home_raw),
                "draw": float(p_draw_raw),
                "away": float(p_away_raw),
            }
        else:
            # Proxy based on predicted direction
            prob_map = {"home": 0.45, "draw": 0.27, "away": 0.28}
            if pred_outcome == "home":
                prob_map["home"] = 0.55
            elif pred_outcome == "away":
                prob_map["away"] = 0.55

        p_for_actual = prob_map[actual_outcome]
        y            = 1 if pred_outcome == actual_outcome else 0
        samples.append((p_for_actual, y))

    n = len(samples)
    if n < MIN_SAMPLES:
        return ProbabilityCalibrator(temperature=1.0, n_samples=n)

    # ── Fit T via scipy minimise_scalar ──────────────────────────────────────
    try:
        from scipy.optimize import minimize_scalar as _ms

        def _nll(T: float) -> float:
            """Negative log-likelihood (binary cross-entropy) over all samples."""
            loss = 0.0
            for p_raw, y in samples:
                p_cal = _temp_scale(p_raw, T)
                p_cal = max(_EPS, min(1.0 - _EPS, p_cal))
                loss -= y * math.log(p_cal) + (1 - y) * math.log(1.0 - p_cal)
            return loss

        res = _ms(_nll, bounds=_T_BOUNDS, method="bounded")
        T   = float(res.x)
        return ProbabilityCalibrator(temperature=round(T, 4), n_samples=n)

    except Exception:
        return ProbabilityCalibrator(temperature=1.0, n_samples=n)
