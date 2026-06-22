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

from core.poisson_engine import ScoreProb
from core.strategy_advisor import StrategyRecommendation, Strategy, TournamentContext
from core.kelly import BetAnalysis


# ---------------------------------------------------------------------------
# Country flag emoji lookup (keyed by lowercase team name)
# ---------------------------------------------------------------------------

_FLAGS: dict[str, str] = {
    # North & Central America
    "united states":          "🇺🇸",
    "usa":                    "🇺🇸",
    "mexico":                 "🇲🇽",
    "canada":                 "🇨🇦",
    "honduras":               "🇭🇳",
    "panama":                 "🇵🇦",
    "costa rica":             "🇨🇷",
    "jamaica":                "🇯🇲",
    "guatemala":              "🇬🇹",
    "el salvador":            "🇸🇻",
    "trinidad and tobago":    "🇹🇹",
    # South America
    "brazil":                 "🇧🇷",
    "argentina":              "🇦🇷",
    "uruguay":                "🇺🇾",
    "colombia":               "🇨🇴",
    "chile":                  "🇨🇱",
    "ecuador":                "🇪🇨",
    "peru":                   "🇵🇪",
    "venezuela":              "🇻🇪",
    "paraguay":               "🇵🇾",
    "bolivia":                "🇧🇴",
    # Europe
    "spain":                  "🇪🇸",
    "france":                 "🇫🇷",
    "germany":                "🇩🇪",
    "england":                "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "portugal":               "🇵🇹",
    "netherlands":            "🇳🇱",
    "belgium":                "🇧🇪",
    "croatia":                "🇭🇷",
    "denmark":                "🇩🇰",
    "switzerland":            "🇨🇭",
    "austria":                "🇦🇹",
    "serbia":                 "🇷🇸",
    "poland":                 "🇵🇱",
    "ukraine":                "🇺🇦",
    "hungary":                "🇭🇺",
    "romania":                "🇷🇴",
    "czech republic":         "🇨🇿",
    "czechia":                "🇨🇿",
    "slovakia":               "🇸🇰",
    "albania":                "🇦🇱",
    "slovenia":               "🇸🇮",
    "turkey":                 "🇹🇷",
    "scotland":               "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "wales":                  "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "north macedonia":        "🇲🇰",
    "bosnia and herzegovina": "🇧🇦",
    "greece":                 "🇬🇷",
    "norway":                 "🇳🇴",
    "sweden":                 "🇸🇪",
    "finland":                "🇫🇮",
    "iceland":                "🇮🇸",
    "georgia":                "🇬🇪",
    # Asia / Oceania
    "japan":                  "🇯🇵",
    "korea republic":         "🇰🇷",
    "south korea":            "🇰🇷",
    "australia":              "🇦🇺",
    "saudi arabia":           "🇸🇦",
    "iran":                   "🇮🇷",
    "qatar":                  "🇶🇦",
    "iraq":                   "🇮🇶",
    "jordan":                 "🇯🇴",
    "oman":                   "🇴🇲",
    "bahrain":                "🇧🇭",
    "uzbekistan":             "🇺🇿",
    "new zealand":            "🇳🇿",
    "indonesia":              "🇮🇩",
    # Africa
    "morocco":                "🇲🇦",
    "senegal":                "🇸🇳",
    "nigeria":                "🇳🇬",
    "ghana":                  "🇬🇭",
    "ivory coast":            "🇨🇮",
    "cameroon":               "🇨🇲",
    "egypt":                  "🇪🇬",
    "algeria":                "🇩🇿",
    "tunisia":                "🇹🇳",
    "mali":                   "🇲🇱",
    "angola":                 "🇦🇴",
    "south africa":           "🇿🇦",
    "dr congo":               "🇨🇩",
    "cape verde islands":     "🇨🇻",
    "cabo verde":             "🇨🇻",
    "tanzania":               "🇹🇿",
    "zambia":                 "🇿🇲",
    "mozambique":             "🇲🇿",
    "benin":                  "🇧🇯",
}


def _flag(team_name: str) -> str:
    """Return the flag emoji for a team, or empty string if unknown."""
    return _FLAGS.get(team_name.lower(), "")


def _with_flag(team_name: str) -> str:
    """Return 'FLAG TeamName' if a flag is known, else just 'TeamName'."""
    flag = _flag(team_name)
    return f"{flag} {team_name}" if flag else team_name


# ---------------------------------------------------------------------------
# Score description
# ---------------------------------------------------------------------------

def _describe_score(home_team: str, away_team: str, home_goals: int, away_goals: int) -> str:
    """
    Return a fully LTR-safe, unambiguous score description.

    Root cause of the inversion bug: inserting a Hebrew word ("מנצחת") between
    English text and digits forces WhatsApp's bidi renderer into mixed-direction
    mode, which can reorder the digits visually (3-0 → 0-3). Using the English
    word "wins" keeps the entire phrase in a single LTR flow, which is immune
    to bidi reordering regardless of surrounding Hebrew context.

    Score is always shown as WINNER_GOALS-LOSER_GOALS (high first), so the
    number itself is unambiguous even if the winner name is read right-to-left
    by a human — the bigger number always belongs to the named team.

    Examples:
      home=Spain 3, away=Saudi Arabia 0  →  "Spain wins 3-0"
      home=Spain 0, away=Saudi Arabia 3  →  "Saudi Arabia wins 3-0"
      home=Spain 1, away=Saudi Arabia 1  →  "Draw 1-1"
    """
    if home_goals > away_goals:
        return f"{home_team} wins {home_goals}-{away_goals}"
    elif away_goals > home_goals:
        return f"{away_team} wins {away_goals}-{home_goals}"
    else:
        return f"Draw {home_goals}-{away_goals}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class DailyPick:
    home_team:      str
    away_team:      str
    recommendation: StrategyRecommendation
    ai_pick:        Optional[ScoreProb]       = None
    ai_reasoning:   Optional[str]             = None
    value_bets:     Optional[list[BetAnalysis]] = None


def format_daily_message(picks: list[DailyPick], context: TournamentContext, perf_report: Optional[dict] = None) -> str:
    """
    Pure function — build the WhatsApp message string.
    Safe to call in tests with assert, no network involved.

    Example output:
      ⚽ *תחזית מונדיאל שטראוס - היום*
      📊 מצב נוכחי: 22 נק' (אתה) | 33 נק' (מוביל)
      📉 פער: 11 נק' | 4 משחקים נותרו

      🎲 *🇪🇸 Spain נגד 🇸🇦 Saudi Arabia*
         ניחוש: *Spain wins 3-0* (11% סיכוי)
         אסטרטגיה: קונטרארי
         (קונצנזוס היה: Spain wins 2-0)

      _נשלח אוטומטית ע"י Mondial Predictor_
    """
    lines = [
        "⚽ *תחזית מונדיאל שטראוס - היום*",
        f"📊 מצב נוכחי: {context.my_points} נק' (אתה) | {context.leader_points} נק' (מוביל)",
        f"📉 פער: {context.point_gap} נק' | {context.matches_remaining} משחקים נותרו",
        "",
    ]

    if perf_report:
        date_label   = perf_report.get("date_label", "")
        correct      = perf_report.get("correct", 0)
        total        = perf_report.get("total", 0)
        exact        = perf_report.get("exact", 0)
        pts_earned   = perf_report.get("pts_earned", 0)
        pts_possible = perf_report.get("pts_possible", 0)
        lines.append(f"📈 *ביצועי אתמול ({date_label})*")
        lines.append(f"   ✅ תוצאה נכונה: {correct}/{total}")
        lines.append(f"   🎯 ניחוש מדויק: {exact}/{total} | {pts_earned}/{pts_possible} נק'")
        lines.append("")

    for pick in picks:
        rec  = pick.recommendation
        icon = "🛡️" if rec.strategy == Strategy.SAFE else "🎲"

        # Use AI ensemble pick when available; fall back to strategy-advisor pick
        final_score = pick.ai_pick if pick.ai_pick else rec.recommended_pick
        pick_desc = _describe_score(
            pick.home_team, pick.away_team,
            final_score.home_goals, final_score.away_goals,
        )
        home_label = _with_flag(pick.home_team)
        away_label = _with_flag(pick.away_team)

        lines.append(f"{icon} *{home_label} נגד {away_label}*")
        lines.append(
            f"   ניחוש: *{pick_desc}* ({final_score.probability * 100:.0f}% סיכוי)"
        )
        lines.append(f"   אסטרטגיה: {rec.strategy.value}")

        if pick.ai_reasoning:
            lines.append(f"   🤖 AI: {pick.ai_reasoning}")

        if rec.strategy == Strategy.CONTRARIAN:
            safe = rec.alternative_safe_pick
            consensus_desc = _describe_score(
                pick.home_team, pick.away_team,
                safe.home_goals, safe.away_goals,
            )
            lines.append(f"   (קונצנזוס היה: {consensus_desc})")

        lines.append("")

    # ── Value Bets section (only when at least one value bet exists) ─────────
    all_value_bets: list[tuple[str, str, BetAnalysis]] = [
        (pick.home_team, pick.away_team, vb)
        for pick in picks
        if pick.value_bets
        for vb in pick.value_bets
    ]
    if all_value_bets:
        lines.append("💰 *הימורי ערך — יתרון מעל 10%*")
        for home, away, vb in all_value_bets:
            outcome_he = {"Home Win": f"ניצחון {home}", "Draw": "תיקו", "Away Win": f"ניצחון {away}"}.get(vb.outcome, vb.outcome)
            lines.append(
                f"   ✨ {_with_flag(home)} נגד {_with_flag(away)} — {outcome_he}"
            )
            lines.append(
                f"      אודס: {vb.decimal_odds:.2f} | "
                f"Edge: {vb.edge_pct:+.1f}% | "
                f"EV: {vb.ev_per_unit:+.1%} | "
                f"Half-Kelly: {vb.half_kelly:.1%} מהבנק"
            )
        lines.append("   ⚠️ _ניתוח מתמטי בלבד — הימרו באחריות_")
        lines.append("")

    lines.append('_נשלח אוטומטית ע"י Mondial Predictor_')
    return "\n".join(lines)


def format_lineup_alert(
    home_team: str,
    away_team: str,
    old_home:  int,
    old_away:  int,
    new_home:  int,
    new_away:  int,
    reasoning: str,
) -> str:
    """
    Build a WhatsApp alert sent when confirmed lineups change the AI prediction.
    Called by run_lineup_check_pipeline() only when the score prediction changed.
    """
    home_label = _with_flag(home_team)
    away_label = _with_flag(away_team)
    old_desc   = _describe_score(home_team, away_team, old_home, old_away)
    new_desc   = _describe_score(home_team, away_team, new_home, new_away)
    return "\n".join([
        "⚠️ *עדכון אסטרטגיה דחוף!*",
        f"📋 {home_label} נגד {away_label}",
        "הסגלים הרשמיים שינו את תחזית הבינה המלאכותית!",
        "",
        "🔄 הניחוש שונה:",
        f"   לשעבר: {old_desc}",
        f"   חדש: *{new_desc}*",
        "",
        f"🤖 {reasoning}",
        "",
        "*עדכן את ההימור שלך בהתאם!*",
    ])


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
