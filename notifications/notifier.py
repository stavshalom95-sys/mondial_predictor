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
from core.kelly import BetAnalysis, Ticket, ConfidenceLeg, ConfidenceTicket

try:
    from core.market_calculator import MarketResult as _MarketResult
except ImportError:
    _MarketResult = None   # type: ignore[assignment,misc]


# ── Budget configuration ──────────────────────────────────────────────────────
# Edit these two values to match your situation.
TOTAL_BANKROLL      = 10_000   # NIS — your full betting bankroll
DAILY_BUDGET_CAP    =    100   # NIS — cap for high-confidence bets (model prob ≥ 40%)
LOW_PROB_BUDGET_CAP =     20   # NIS — cap for low-confidence bets (model prob < 40%)


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
    ai_pick:        Optional[ScoreProb]         = None
    ai_reasoning:   Optional[str]               = None
    value_bets:     Optional[list[BetAnalysis]] = None
    market_data:            Optional[object]       = None   # MarketResult | None
    sg_value_bet:           Optional[str]         = None   # "0-1" | "2-3" | "+4" | None
    tournament_context_lines: Optional[list[str]] = None   # pre-formatted WhatsApp lines
    logic_chain:    Optional[str]               = None   # λ adjustment chain for transparency
    # ── Simulation output (drives final prediction) ──────────────────────────
    sim_score_home: Optional[int]   = None   # most likely score from Poisson score matrix
    sim_score_away: Optional[int]   = None
    sim_p_home:     Optional[float] = None   # MC win-rate (10k draws)
    sim_p_draw:     Optional[float] = None
    sim_p_away:     Optional[float] = None
    poisson_p_home: Optional[float] = None   # analytical Poisson probability
    poisson_p_draw: Optional[float] = None
    poisson_p_away: Optional[float] = None
    why_bullets:    Optional[list[str]] = None  # 3-5 bullets explaining the prediction


def format_daily_message(
    picks:       list[DailyPick],
    context:     TournamentContext,
    perf_report: Optional[dict]             = None,
    ticket:      Optional[Ticket]           = None,
    prob_ticket: Optional[Ticket]           = None,
    conf_ticket: Optional[ConfidenceTicket] = None,
) -> str:
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
    ]
    if getattr(context, "standings_source", "fallback") == "fallback":
        lines.append("⚠️ _דירוגים: נתוני גיבוי (365Scores לא זמין) — פער עשוי להיות לא מעודכן_")
    lines += [
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
        pnl_nis      = perf_report.get("pnl_nis")
        bets_placed  = perf_report.get("bets_placed", 0)
        at_correct   = perf_report.get("all_time_correct")
        at_total     = perf_report.get("all_time_total", 0)
        at_hit_rate  = perf_report.get("all_time_hit_rate")
        at_pnl       = perf_report.get("all_time_pnl")

        if total > 0:
            lines.append(f"📈 *ביצועי אתמול ({date_label})*")
            lines.append(f"   ✅ תוצאה נכונה: {correct}/{total}")
            lines.append(f"   🎯 ניחוש מדויק: {exact}/{total} | {pts_earned}/{pts_possible} נק'")
            if pnl_nis is not None and bets_placed:
                _sign = "+" if pnl_nis >= 0 else ""
                _plural = "ים" if bets_placed > 1 else ""
                lines.append(f"   💰 P&L הימורים: {_sign}{pnl_nis:.0f} ₪ ({bets_placed} הימור{_plural})")
        if at_total and at_total > 0:
            _hr_pct = round(at_hit_rate * 100) if at_hit_rate else 0
            lines.append(f"   📊 Hit Rate כללי: {_hr_pct}% ({at_correct}/{at_total})")
            if at_pnl is not None:
                _sign2 = "+" if at_pnl >= 0 else ""
                lines.append(f"   💹 P&L מצטבר: {_sign2}{at_pnl:.0f} ₪")
        lines.append("")

    for pick in picks:
        rec  = pick.recommendation
        icon = "🛡️" if rec.strategy == Strategy.SAFE else "🎲"
        home_label = _with_flag(pick.home_team)
        away_label = _with_flag(pick.away_team)

        lines.append(f"{icon} *{home_label} נגד {away_label}*")

        # ── Simulation block (primary output) ────────────────────────────────
        has_sim = pick.sim_score_home is not None and pick.sim_score_away is not None
        if pick.poisson_p_home is not None:
            lines.append(
                f"   📊 Poisson: H={pick.poisson_p_home:.0%}  "
                f"D={pick.poisson_p_draw:.0%}  A={pick.poisson_p_away:.0%}"
            )
        if pick.sim_p_home is not None:
            lines.append(
                f"   🎲 Sim (10k): H={pick.sim_p_home:.0%}  "
                f"D={pick.sim_p_draw:.0%}  A={pick.sim_p_away:.0%}"
            )

        # Final prediction = simulation's most likely score; fall back to AI pick, then strategy advisor
        if has_sim:
            pick_desc = _describe_score(
                pick.home_team, pick.away_team,
                pick.sim_score_home, pick.sim_score_away,
            )
        else:
            _fs = pick.ai_pick if pick.ai_pick else rec.recommended_pick
            pick_desc = _describe_score(
                pick.home_team, pick.away_team,
                _fs.home_goals, _fs.away_goals,
            )

        lines.append(f"   ⚽ *Final Prediction: {pick_desc}*")
        lines.append(f"   אסטרטגיה: {rec.strategy.value}")

        # ── Why section ───────────────────────────────────────────────────────
        if pick.why_bullets:
            lines.append("   📌 *למה?*")
            for _b in pick.why_bullets:
                lines.append(f"      {_b}")

        # ── Supplementary context (ESPN / RapidAPI reasoning) ────────────────
        if pick.ai_reasoning:
            lines.append(f"   💡 Context: {pick.ai_reasoning}")
        if pick.logic_chain:
            lines.append(f"   📐 Chain: {pick.logic_chain}")

        if rec.strategy == Strategy.CONTRARIAN:
            safe = rec.alternative_safe_pick
            consensus_desc = _describe_score(
                pick.home_team, pick.away_team,
                safe.home_goals, safe.away_goals,
            )
            lines.append(f"   (קונצנזוס היה: {consensus_desc})")

        # ── Tournament context (rotation / motivation) ────────────────────
        if pick.tournament_context_lines:
            for _tcl in pick.tournament_context_lines:
                lines.append(_tcl)

        # ── Sub-markets block ─────────────────────────────────────────────
        if pick.market_data is not None:
            m = pick.market_data

            # Outcome probabilities from goal_diff distribution
            gd    = m.goal_diff
            p_h   = round(sum(v for k, v in gd.items() if k > 0), 3)
            p_d   = round(gd.get(0, 0.0), 3)
            p_a   = round(sum(v for k, v in gd.items() if k < 0), 3)
            if p_h >= p_d and p_h >= p_a:
                winner_label, winner_pct = pick.home_team, p_h
            elif p_d >= p_h and p_d >= p_a:
                winner_label, winner_pct = "Draw", p_d
            else:
                winner_label, winner_pct = pick.away_team, p_a

            # Sum-goals 3-way bracket
            sg = m.sum_goals if hasattr(m, "sum_goals") else {}

            # Asian handicap -1.5 for home team
            ah = next((h for h in m.handicaps if h.get("handicap") == -1.5), None)

            btts = m.btts

            lines.append("   📊 *Markets:*")
            lines.append(f"      🏆 Winner: {winner_label} ({winner_pct:.1%})")
            if sg:
                _sg_badge = f" 🔥 VALUE ({pick.sg_value_bet})" if pick.sg_value_bet else ""
                lines.append(
                    f"      📈 Goals: 0-1 ({sg.get('0-1', 0):.0%})"
                    f" | 2-3 ({sg.get('2-3', 0):.0%})"
                    f" | 4+ ({sg.get('+4', 0):.0%}){_sg_badge}"
                )
            if ah:
                lines.append(
                    f"      ⚖️  AH (-1.5): {pick.home_team} covers "
                    f"({ah['p_home_covers']:.1%})"
                )
            if btts:
                lines.append(f"      ⚽ BTTS: Yes ({btts['p_yes']:.1%})")

        lines.append("")

    # ── Value Bets section (Value = model_prob × odds > 1.05) ────────────────
    all_value_bets: list[tuple[str, str, BetAnalysis]] = [
        (pick.home_team, pick.away_team, vb)
        for pick in picks
        if pick.value_bets
        for vb in pick.value_bets
        if vb.is_value
    ]
    if all_value_bets:
        lines.append("💰 *VALUE BETS — ניתוח מתמטי (Value > 1.05)*")
        for home, away, vb in all_value_bets:
            outcome_he = {
                "Home Win": f"ניצחון {home}",
                "Draw":     "תיקו",
                "Away Win": f"ניצחון {away}",
            }.get(vb.outcome, vb.outcome)
            lines.append(f"   ✨ {_with_flag(home)} נגד {_with_flag(away)} — {outcome_he}")

            # Transparency math block
            lines.append(
                f"      📐 Prob {vb.our_prob:.0%} × Odds {vb.decimal_odds:.2f}"
                f" = Value {vb.value:.3f} | Edge {vb.edge_pct:+.1f}%"
            )

            # Kelly stake
            _stake_raw = vb.half_kelly * TOTAL_BANKROLL
            if vb.our_prob >= 0.40:
                _capped = _stake_raw > DAILY_BUDGET_CAP
                _stake  = DAILY_BUDGET_CAP if _capped else _stake_raw
                _cap_tag = " ⚠️ (מקסימום יומי)" if _capped else ""
            else:
                _stake_raw = _stake_raw / 4
                _capped = _stake_raw > LOW_PROB_BUDGET_CAP
                _stake  = LOW_PROB_BUDGET_CAP if _capped else _stake_raw
                _cap_tag = " ⚠️ (הגבלת סיכון)" if _capped else " (÷4 — סיכוי נמוך)"
            lines.append(f"      💰 הימור מומלץ: {_stake:.0f} ₪{_cap_tag}")

        lines.append("   ⚠️ _ניתוח מתמטי בלבד — הימרו באחריות_")
        lines.append("")

    # ── Value Ticket (EV-based) ───────────────────────────────────────────────
    if ticket and len(ticket.legs) >= 2:
        ticket_type = {2: "Double 🎯", 3: "Triple 🔥"}.get(len(ticket.legs), "Parlay")
        lines.append(f"🎟️ *VALUE TICKET — {ticket_type}*")
        for i, leg in enumerate(ticket.legs, 1):
            outcome_short = {"Home Win": "Win", "Draw": "Draw", "Away Win": "Win"}.get(leg.outcome, leg.outcome)
            side = leg.match_label.split(" vs ")[0] if leg.outcome == "Home Win" else (
                leg.match_label.split(" vs ")[1] if leg.outcome == "Away Win" else "Draw"
            )
            lines.append(
                f"   {i}. {leg.match_label} — {side} ({outcome_short})"
                f"  Odds {leg.decimal_odds:.2f}  [Value {leg.value:.3f}]"
            )
        lines.append(
            f"   📐 Combined odds: {ticket.combined_odds:.2f} | "
            f"Model prob: {ticket.combined_prob:.1%} | "
            f"EV: {ticket.ev_combined:+.1%}"
        )
        lines.append(f"   💰 Recommended stake: {ticket.stake_nis:.0f} ₪")
        lines.append(f"   🏆 Potential return: {ticket.stake_nis * ticket.combined_odds:.0f} ₪")
        lines.append("")

    # ── Probability Ticket (Straight Wins, Sim ≥ 65%) ────────────────────────
    if prob_ticket and len(prob_ticket.legs) >= 2:
        pt_type = {2: "Double 🎯", 3: "Triple 🔥"}.get(len(prob_ticket.legs), "Parlay")
        lines.append(f"🏆 *HIGH-PROBABILITY TICKET — Straight Wins {pt_type}*")
        lines.append(f"   _(מסוננים לפי הסתברות ניצחון Sim ≥ 65% בלבד)_")
        for i, leg in enumerate(prob_ticket.legs, 1):
            side = (
                leg.match_label.split(" vs ")[0] if leg.outcome == "Home Win"
                else leg.match_label.split(" vs ")[1]
            )
            lines.append(
                f"   {i}. {leg.match_label} — {_with_flag(side.strip())} Win"
                f"  Odds {leg.decimal_odds:.2f}  [Sim {leg.our_prob:.0%}]"
            )
        lines.append(
            f"   📐 Combined probability: {prob_ticket.combined_prob:.1%} | "
            f"Combined odds: {prob_ticket.combined_odds:.2f}"
        )
        lines.append(f"   💰 Recommended stake: {prob_ticket.stake_nis:.0f} ₪")
        lines.append(f"   🏆 Potential return: {prob_ticket.stake_nis * prob_ticket.combined_odds:.0f} ₪")
        lines.append("")

    # ── Confidence Value Ticket (Sim ≥ 60% + edge ≥ 5% on Winner) ───────────
    if conf_ticket and len(conf_ticket.legs) >= 2:
        ct_type = {2: "Double 🎯", 3: "Triple 🔥"}.get(len(conf_ticket.legs), "Parlay")
        lines.append(f"💎 *CONFIDENCE VALUE TICKET — {ct_type}*")
        lines.append(f"   _(Sim ≥ 60% + Model edge ≥ 5% מול המסחר — ניצחונות בלבד)_")
        for i, leg in enumerate(conf_ticket.legs, 1):
            lines.append(
                f"   {i}. {leg.match_label} — {_with_flag(leg.winner_name.strip())} Win"
                f"  Odds {leg.decimal_odds:.2f}"
            )
            lines.append(
                f"      📐 Model thinks {leg.sim_prob:.0%}, "
                f"Market thinks {leg.implied_prob:.0%}. "
                f"Value Edge: +{leg.edge:.0%}"
            )
            lines.append(f"      EV: {leg.ev:+.1%}")
        lines.append(
            f"   📊 Combined odds: {conf_ticket.combined_odds:.2f} | "
            f"Combined prob: {conf_ticket.combined_prob:.1%} | "
            f"Total EV: {conf_ticket.total_ev:+.1%}"
        )
        lines.append(f"   💰 Recommended stake: {conf_ticket.stake_nis:.0f} ₪")
        lines.append(f"   🏆 Potential return: {conf_ticket.stake_nis * conf_ticket.combined_odds:.0f} ₪")
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

    import time
    _delays = [0, 3, 5]   # wait 0s before attempt 1, 3s before attempt 2, 5s before attempt 3
    for _attempt, _wait in enumerate(_delays, 1):
        if _wait:
            time.sleep(_wait)
        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            print(f"[notifier] WhatsApp message sent successfully (attempt {_attempt}).")
            return True
        except Exception as exc:
            print(f"[notifier] WhatsApp send failed (attempt {_attempt}/3): {exc}")

    # All attempts exhausted — print to stdout so the pick is not lost
    print("[notifier] All 3 send attempts failed. Message content:")
    print(message)
    return False
