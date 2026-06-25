"""
ai_ensemble.py — Claude contextual calibration layer.

Takes the Poisson top-3 candidates + live match context (injuries, form, lineups)
and returns the best exact score prediction with a brief reasoning note.

Model  : claude-opus-4-6
Output : Pydantic-validated structured output via `instructor` (auto-retry on
         malformed JSON, type coercion, validation error feedback to model).
         Falls back to raw Anthropic SDK if instructor not installed.
Cost   : ~$0.01–0.03 per match call with Opus 4.6.

AI Research Skills applied:
  - instructor skill (16-prompt-engineering/instructor): Pydantic + auto-retry
  - brainstorming-research-ideas Framework 6 (Failure Analysis):
      previous output_config+thinking approach failed silently on schema mismatches;
      instructor fixes this with max_retries=2 + validation error feedback.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

from config.scoring_rules import TournamentStage
from core.poisson_engine import PoissonMatchModel, ScoreProb

# ── Optional dependency checks ───────────────────────────────────────────────
try:
    import anthropic as _anthropic
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

try:
    import instructor as _instructor
    from pydantic import BaseModel, Field
    _INSTRUCTOR_AVAILABLE = True
except ImportError:
    _INSTRUCTOR_AVAILABLE = False

_MODEL = "claude-opus-4-6"

_SYSTEM_PROMPT = """\
You are an expert football analyst assisting in a World Cup prediction competition.
Your job: given Poisson statistical probabilities derived from betting market odds,
and live pre-match context (injuries, recent form, confirmed lineups if available),
recommend the single best exact score to predict.

Guidelines:
- Strongly prefer scores from the provided Poisson top-3 candidates.
- You may deviate by ±1 goal if contextual evidence (key player absent, exceptional form) clearly warrants it.
- home_goals and away_goals must each be integers between 0 and 5.
- Keep reasoning concise (1–2 sentences) — it appears in a WhatsApp prediction message.
- Be direct: state whether context reinforces or challenges the statistical baseline.
"""

# ── Pydantic schema (instructor-validated) ───────────────────────────────────
if _INSTRUCTOR_AVAILABLE:
    class _EnsembleSchema(BaseModel):
        chosen_home_goals: int = Field(..., ge=0, le=5, description="Predicted home team goals (0-5)")
        chosen_away_goals: int = Field(..., ge=0, le=5, description="Predicted away team goals (0-5)")
        reasoning:         str = Field(..., description="1-2 sentence explanation for the WhatsApp message")
        confidence_level:  Literal["high", "medium", "low"] = Field(..., description="How much context shifts the pick vs. pure statistics")
        overrode_poisson:  bool = Field(..., description="True if the chosen score differs from the Poisson #1 pick")


# ── Public data class ────────────────────────────────────────────────────────

@dataclass
class EnsemblePick:
    chosen_home_goals: int
    chosen_away_goals: int
    reasoning:         str
    confidence_level:  str    # "high" | "medium" | "low"
    overrode_poisson:  bool

    def to_score_prob(self, model: PoissonMatchModel) -> ScoreProb:
        prob = model.probability_of(self.chosen_home_goals, self.chosen_away_goals)
        return ScoreProb(
            home_goals  = self.chosen_home_goals,
            away_goals  = self.chosen_away_goals,
            probability = prob,
        )


# ── Private helpers ──────────────────────────────────────────────────────────

_HIGH_VALUE_THRESHOLD = 0.20


def _build_user_prompt(
    home_team:                  str,
    away_team:                  str,
    stage:                      TournamentStage,
    model:                      PoissonMatchModel,
    context_section:            str,
    value_bet_edge:             float = 0.0,
    value_bet_outcome:          str   = "",
    tournament_context_section: str   = "",
) -> str:
    top3 = model.top_n(3)
    candidates = "\n".join(
        f"  {i + 1}. {c.home_goals}-{c.away_goals} ({c.probability * 100:.1f}%)"
        for i, c in enumerate(top3)
    )

    parts: list[str] = []
    if tournament_context_section.strip():
        parts.append(tournament_context_section.strip())
    if context_section.strip():
        parts.append(context_section.strip())
    context_block = "\n\n".join(parts) if parts else (
        "No live context available — base your decision on statistics only."
    )

    value_alert = ""
    if value_bet_edge >= _HIGH_VALUE_THRESHOLD:
        value_alert = (
            f"\n⚠️  VALUE ALERT — Monte Carlo edge: {value_bet_edge:+.1%} on {value_bet_outcome.upper()} outcome.\n"
            f"The simulation's implied probability is {value_bet_edge:.0%} higher than the bookmaker's "
            f"implied probability for {value_bet_outcome}. This is a statistically significant mispricing.\n"
            f"REQUIREMENT: Your reasoning MUST explicitly state that the statistical edge (value) "
            f"outweighs the default Poisson exact-score bias toward low-scoring draws. "
            f"Explain why backing the {value_bet_outcome.upper()} side is the rational choice "
            f"given this {value_bet_edge:.0%} edge. Do NOT default to a conservative draw prediction "
            f"if the edge points clearly to {value_bet_outcome}.\n"
        )

    return (
        f"Match: {home_team} vs {away_team}\n"
        f"Stage: {stage.value}\n\n"
        f"Poisson statistical model (calibrated from market odds):\n"
        f"  Expected goals — {home_team}: {model.lambda_home:.2f} | {away_team}: {model.lambda_away:.2f}\n"
        f"  Top-3 most likely exact scores:\n{candidates}\n\n"
        f"Live pre-match context:\n{context_block}\n"
        f"{value_alert}\n"
        f"Based on the above, choose the best exact score prediction. "
        f"Explain briefly whether the context reinforces or changes the statistical outlook."
    )


def _pick_from_dict(data: dict) -> EnsemblePick:
    return EnsemblePick(
        chosen_home_goals = int(data["chosen_home_goals"]),
        chosen_away_goals = int(data["chosen_away_goals"]),
        reasoning         = str(data["reasoning"]),
        confidence_level  = str(data.get("confidence_level", "medium")),
        overrode_poisson  = bool(data.get("overrode_poisson", False)),
    )


# ── Public API ───────────────────────────────────────────────────────────────

def enhance(
    home_team:                  str,
    away_team:                  str,
    stage:                      TournamentStage,
    model:                      PoissonMatchModel,
    context_section:            str          = "",
    api_key:                    Optional[str] = None,
    value_bet_edge:             float        = 0.0,
    value_bet_outcome:          str          = "",
    tournament_context_section: str          = "",
) -> Optional[EnsemblePick]:
    """
    Call Claude to select the best exact score given Poisson stats + live context.

    Uses instructor (Pydantic + auto-retry) when available, falls back to
    raw Anthropic SDK otherwise. Returns None on any failure — caller falls
    back to Poisson #1.
    """
    if not _SDK_AVAILABLE:
        print("[ensemble] 'anthropic' package not installed — skipping AI ensemble.")
        return None

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[ensemble] ANTHROPIC_API_KEY not set — skipping AI ensemble.")
        return None

    if value_bet_edge >= _HIGH_VALUE_THRESHOLD:
        print(
            f"[ensemble] ⭐ HIGH-VALUE ALERT: {value_bet_edge:+.1%} edge on {value_bet_outcome} — "
            f"injecting value-priority directive into prompt."
        )

    prompt   = _build_user_prompt(
        home_team, away_team, stage, model, context_section,
        value_bet_edge, value_bet_outcome, tournament_context_section,
    )
    messages = [{"role": "user", "content": prompt}]
    print(f"[ensemble] Calling {_MODEL} for {home_team} vs {away_team}...")

    try:
        # ── Path A: instructor — Pydantic + auto-retry (preferred) ───────────
        if _INSTRUCTOR_AVAILABLE:
            client = _instructor.from_anthropic(_anthropic.Anthropic(api_key=api_key))
            raw    = client.messages.create(
                model          = _MODEL,
                max_tokens     = 1024,
                system         = _SYSTEM_PROMPT,
                messages       = messages,
                response_model = _EnsembleSchema,
                max_retries    = 2,
            )
            pick = EnsemblePick(
                chosen_home_goals = raw.chosen_home_goals,
                chosen_away_goals = raw.chosen_away_goals,
                reasoning         = raw.reasoning,
                confidence_level  = raw.confidence_level,
                overrode_poisson  = raw.overrode_poisson,
            )

        # ── Path B: raw SDK — simple text response + JSON parse ───────────────
        else:
            import json
            print("[ensemble] instructor not installed — using raw SDK fallback.")
            client   = _anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model    = _MODEL,
                max_tokens = 1024,
                system   = _SYSTEM_PROMPT,
                messages = messages,
            )
            text_block = next((b for b in response.content if b.type == "text"), None)
            if text_block is None:
                print("[ensemble] No text block in response — falling back to Poisson.")
                return None
            # Extract JSON from response (may be wrapped in markdown code fence)
            text = text_block.text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            pick = _pick_from_dict(json.loads(text.strip()))

        override_note = " [DEVIATED from Poisson #1]" if pick.overrode_poisson else ""
        print(
            f"[ensemble]   Pick: {pick.chosen_home_goals}-{pick.chosen_away_goals}"
            f" ({pick.confidence_level} confidence){override_note}"
        )
        print(f"[ensemble]   Reasoning: {pick.reasoning}")
        return pick

    except _anthropic.AuthenticationError:
        print("[ensemble] Invalid ANTHROPIC_API_KEY — check your secret.")
        return None
    except _anthropic.RateLimitError:
        print("[ensemble] Anthropic rate limit hit — falling back to Poisson.")
        return None
    except Exception as exc:
        print(f"[ensemble] Claude call failed: {exc}")
        return None
