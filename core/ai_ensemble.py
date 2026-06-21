"""
ai_ensemble.py — Claude contextual calibration layer.

Takes the Poisson top-3 candidates + live match context (injuries, form, lineups)
and returns the best exact score prediction with a brief reasoning note.

Model  : claude-opus-4-6 with adaptive thinking.
Output : structured JSON via output_config (Pydantic-compatible schema).
Cost   : ~$0.01–0.03 per match call with Opus 4.6 (well within budget).

New env var / GitHub Secret: ANTHROPIC_API_KEY
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from config.scoring_rules import TournamentStage
from core.poisson_engine import PoissonMatchModel, ScoreProb

try:
    import anthropic as _anthropic
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

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

# JSON schema for structured output (additionalProperties: false required by API)
_OUTPUT_SCHEMA: dict = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "chosen_home_goals": {
                "type": "integer",
                "description": "Predicted home team goals (0–5)",
            },
            "chosen_away_goals": {
                "type": "integer",
                "description": "Predicted away team goals (0–5)",
            },
            "reasoning": {
                "type": "string",
                "description": "1–2 sentence explanation shown to the user in WhatsApp",
            },
            "confidence_level": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "How much context shifts the pick vs. pure statistics",
            },
            "overrode_poisson": {
                "type": "boolean",
                "description": "True if the chosen score differs from the Poisson #1 pick",
            },
        },
        "required": [
            "chosen_home_goals",
            "chosen_away_goals",
            "reasoning",
            "confidence_level",
            "overrode_poisson",
        ],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass
class EnsemblePick:
    chosen_home_goals: int
    chosen_away_goals: int
    reasoning:         str
    confidence_level:  str    # "high" | "medium" | "low"
    overrode_poisson:  bool

    def to_score_prob(self, model: PoissonMatchModel) -> ScoreProb:
        """Convert to a ScoreProb using the probability from the Poisson model."""
        prob = model.probability_of(self.chosen_home_goals, self.chosen_away_goals)
        return ScoreProb(
            home_goals  = self.chosen_home_goals,
            away_goals  = self.chosen_away_goals,
            probability = prob,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_user_prompt(
    home_team:       str,
    away_team:       str,
    stage:           TournamentStage,
    model:           PoissonMatchModel,
    context_section: str,
) -> str:
    top3 = model.top_n(3)
    candidates = "\n".join(
        f"  {i + 1}. {c.home_goals}-{c.away_goals} ({c.probability * 100:.1f}%)"
        for i, c in enumerate(top3)
    )
    context_block = context_section.strip() if context_section.strip() else (
        "No live context available — base your decision on statistics only."
    )
    return (
        f"Match: {home_team} vs {away_team}\n"
        f"Stage: {stage.value}\n\n"
        f"Poisson statistical model (calibrated from market odds):\n"
        f"  Expected goals — {home_team}: {model.lambda_home:.2f} | {away_team}: {model.lambda_away:.2f}\n"
        f"  Top-3 most likely exact scores:\n{candidates}\n\n"
        f"Live pre-match context:\n{context_block}\n\n"
        f"Based on the above, choose the best exact score prediction. "
        f"Explain briefly whether the context reinforces or changes the statistical outlook."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enhance(
    home_team:       str,
    away_team:       str,
    stage:           TournamentStage,
    model:           PoissonMatchModel,
    context_section: str = "",
    api_key:         Optional[str] = None,
) -> Optional[EnsemblePick]:
    """
    Call Claude to select the best exact score given Poisson stats + live context.

    Args:
        home_team: Schedule home team name.
        away_team: Schedule away team name.
        stage: TournamentStage (affects prompt framing).
        model: Calibrated PoissonMatchModel for this match.
        context_section: Text block from MatchContext.to_prompt_section() (may be "").
        api_key: ANTHROPIC_API_KEY. Falls back to env var.

    Returns:
        EnsemblePick with chosen score + reasoning, or None on any failure.
        On failure the caller should fall back to Poisson #1.
    """
    if not _SDK_AVAILABLE:
        print("[ensemble] 'anthropic' package not installed — skipping AI ensemble.")
        return None

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[ensemble] ANTHROPIC_API_KEY not set — skipping AI ensemble.")
        return None

    client = _anthropic.Anthropic(api_key=api_key)
    prompt = _build_user_prompt(home_team, away_team, stage, model, context_section)

    print(f"[ensemble] Calling {_MODEL} for {home_team} vs {away_team}...")

    try:
        response = client.messages.create(
            model        = _MODEL,
            max_tokens   = 1024,
            thinking     = {"type": "adaptive"},
            system       = _SYSTEM_PROMPT,
            messages     = [{"role": "user", "content": prompt}],
            output_config= {"format": _OUTPUT_SCHEMA},
        )

        # Adaptive thinking returns thinking + text blocks; extract the text block.
        text_block = next(
            (b for b in response.content if b.type == "text"), None
        )
        if text_block is None:
            print("[ensemble] No text block in response — falling back to Poisson.")
            return None

        data = json.loads(text_block.text)
        pick = EnsemblePick(
            chosen_home_goals = int(data["chosen_home_goals"]),
            chosen_away_goals = int(data["chosen_away_goals"]),
            reasoning         = str(data["reasoning"]),
            confidence_level  = str(data.get("confidence_level", "medium")),
            overrode_poisson  = bool(data.get("overrode_poisson", False)),
        )

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
