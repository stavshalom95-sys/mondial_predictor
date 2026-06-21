"""
WhatsApp notifier via Green-API.

Two clean-separated functions:
  format_daily_message  — pure function, testable without side effects
  send_whatsapp_message — network side-effect; gracefully degrades to stdout
                          if credentials are missing (never crashes the pipeline)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from core.strategy_advisor import StrategyRecommendation, Strategy, TournamentContext


@dataclass
class DailyPick:
    home_team:      str
    away_team:      str
    recommendation: StrategyRecommendation


def format_daily_message(picks: list[DailyPick], context: TournamentContext) -> str:
    """
    Pure function — build the WhatsApp message string.
    Safe to call in tests with assert, no network involved.

    Example output:
      ⚽ *תחזית מונדיאל שטראוס - היום*
      📊 מצב נוכחי: 22 נק' (אתה) | 33 נק' (מוביל)
      📉 פער: 11 נק' | 2 משחקים נותרו

      🎲 *Spain נגד Saudi Arabia*
         ניחוש: *1:0* (10% סיכוי)
         אסטרטגיה: קונטרארי | שלב: שלב הבתים
         ניקוד: בול=3 | כיוון=1
         (קונצנזוס היה: 2:0)

      _נשלח אוטומטית ע"י Mondial Predictor_
    """
    lines = [
        "⚽ *תחזית מונדיאל שטראוס - היום*",
        f"📊 מצב נוכחי: {context.my_points} נק' (אתה) | {context.leader_points} נק' (מוביל)",
        f"📉 פער: {context.point_gap} נק' | {context.matches_remaining} משחקים נותרו",
        "",
    ]

    for pick in picks:
        rec  = pick.recommendation
        icon = "🛡️" if rec.strategy == Strategy.SAFE else "🎲"

        lines.append(f"{icon} *{pick.home_team} נגד {pick.away_team}*")
        lines.append(
            f"   ניחוש: *{rec.recommended_pick.home_goals}:{rec.recommended_pick.away_goals}* "
            f"({rec.recommended_pick.probability * 100:.0f}% סיכוי)"
        )
        lines.append(
            f"   אסטרטגיה: {rec.strategy.value} | שלב: {rec.stage.value}"
        )
        lines.append(
            f"   ניקוד: בול={rec.points_if_exact} | כיוון={rec.points_if_direction_only}"
        )

        if rec.strategy == Strategy.CONTRARIAN:
            safe = rec.alternative_safe_pick
            lines.append(f"   (קונצנזוס היה: {safe.home_goals}:{safe.away_goals})")

        lines.append("")

    lines.append('_נשלח אוטומטית ע"י Mondial Predictor_')
    return "\n".join(lines)


def send_whatsapp_message(
    message:         str,
    instance_id:     Optional[str] = None,
    api_token:       Optional[str] = None,
    recipient_phone: Optional[str] = None,
) -> bool:
    """
    Send `message` via Green-API.
    Falls back to printing to stdout if any credential is missing — never raises.

    Credentials resolved from args first, then environment variables:
      GREEN_API_INSTANCE_ID
      GREEN_API_TOKEN
      WHATSAPP_RECIPIENT_PHONE  (international format, e.g. 972501234567)
    """
    import requests  # imported here to keep module loadable without requests installed

    instance_id     = instance_id     or os.environ.get("GREEN_API_INSTANCE_ID")
    api_token       = api_token       or os.environ.get("GREEN_API_TOKEN")
    recipient_phone = recipient_phone or os.environ.get("WHATSAPP_RECIPIENT_PHONE")

    if not all([instance_id, api_token, recipient_phone]):
        print("[notifier] WhatsApp credentials missing — printing message to terminal:\n")
        print(message)
        return False

    url = (
        f"https://api.green-api.com/waInstance{instance_id}"
        f"/sendMessage/{api_token}"
    )
    payload = {
        "chatId": f"{recipient_phone}@c.us",
        "message": message,
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        print("[notifier] WhatsApp message sent successfully.")
        return True
    except Exception as exc:
        print(f"[notifier] WhatsApp send failed: {exc}")
        print("[notifier] Message content:\n", message)
        return False
