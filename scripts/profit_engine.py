"""
scripts/profit_engine.py — Sniper Protocol Profit Engine

Reads winner_odds.json from repo root, builds strength model from data/history.json,
runs Monte Carlo simulation (20k draws), and applies the Sniper Protocol filter.

Sniper Protocol:
  Gate 1: EV  > EV_MIN   (default 10%)
  Gate 2: DS  > DS_MIN   (default 70)

Decision Score formula (0–100):
  DS = 100 × (0.50 × min(prob/0.80,1) + 0.30 × min(edge/0.15,1) + 0.20 × min(ev/0.20,1))

Stake sizing: Quarter-Kelly × bankroll (NO external APIs called).

Usage:
  python scripts/profit_engine.py [--bankroll 100] [--ds-min 70] [--ev-min 0.10]
"""
from __future__ import annotations

import sys, json, os, argparse
sys.stdout.reconfigure(encoding="utf-8")

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)

from core.strength_model import build_strength_model
from core.simulator import simulate

# ── Defaults (Sniper Protocol v2) ─────────────────────────────────────────────
_BANKROLL  = 100.0
_KF        = 0.25    # quarter-Kelly
_EV_MIN    = 0.10    # Gate 1: minimum EV
_DS_MIN    = 70.0    # Gate 2: minimum Decision Score  ← updated from 75


def decision_score(prob: float, edge: float, ev: float) -> float:
    """Composite 0–100 score weighting probability, edge, and EV."""
    s_prob = min(prob / 0.80, 1.0)
    s_edge = min(edge / 0.15, 1.0) if edge > 0 else 0.0
    s_ev   = min(ev   / 0.20, 1.0) if ev   > 0 else 0.0
    return round(100 * (0.50 * s_prob + 0.30 * s_edge + 0.20 * s_ev), 1)


def run(bankroll: float = _BANKROLL, ev_min: float = _EV_MIN, ds_min: float = _DS_MIN) -> None:
    # ── Build strength model from tournament history ───────────────────────────
    history_path = os.path.join(_ROOT, "data", "history.json")
    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    results = []
    for g in history:
        ht = g.get("home_team", "")
        at = g.get("away_team", "")
        hg = g.get("actual_home")
        ag = g.get("actual_away")
        if not ht or not at or hg is None or ag is None:
            continue
        try:
            results.append({
                "home_team":  ht,
                "away_team":  at,
                "home_goals": int(hg),
                "away_goals": int(ag),
            })
        except (ValueError, TypeError):
            continue

    sm = build_strength_model(results)
    if sm is None:
        # Fallback: seed with 3 neutral dummy matches so model initialises
        sm = build_strength_model([
            {"home_team": "__A", "away_team": "__B", "home_goals": 1, "away_goals": 1},
            {"home_team": "__C", "away_team": "__D", "home_goals": 2, "away_goals": 0},
            {"home_team": "__E", "away_team": "__F", "home_goals": 0, "away_goals": 1},
        ])
        print("[WARNING] < 3 completed matches in history — using FIFA priors only")
    print(sm.summary())

    # ── Load odds ──────────────────────────────────────────────────────────────
    odds_path = os.path.join(_ROOT, "winner_odds.json")
    with open(odds_path, encoding="utf-8") as f:
        odds_data = json.load(f)

    SKIP_KEYS = {"_note", "_played_today"}
    matches = {k: v for k, v in odds_data.items()
               if k not in SKIP_KEYS and isinstance(v, dict)}

    # ── Header ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  🎯  PROFIT ENGINE — DAILY BETTING PLAN")
    print("=" * 65)
    print(f"  Bankroll: {bankroll:.0f} NIS  |  ¼-Kelly  |  EV≥{ev_min:.0%}  DS≥{ds_min:.0f}")
    print(f"  ⚡ No external APIs called — internal model + history only")
    print("=" * 65)

    sniper_bets: list[dict] = []
    skipped:     list[str]  = []

    for match_label, markets in matches.items():
        winner    = markets.get("winner", {})
        home_odds = winner.get("home", 0.0)
        draw_odds = winner.get("draw", 0.0)
        away_odds = winner.get("away", 0.0)

        if home_odds == 0.0 and draw_odds == 0.0 and away_odds == 0.0:
            skipped.append(match_label)
            continue

        parts = match_label.split(" vs ", 1)
        if len(parts) != 2:
            skipped.append(match_label)
            continue

        home_team, away_team = parts[0].strip(), parts[1].strip()
        lam_h, lam_a = sm.lambdas(home_team, away_team)
        sim_r = simulate(lam_h, lam_a, n_sims=20_000)

        print(f"\n📋 {match_label}")
        print(f"   λ  home={lam_h:.2f}  away={lam_a:.2f}")
        print(f"   Sim  H={sim_r.p_home:.1%}  D={sim_r.p_draw:.1%}  A={sim_r.p_away:.1%}  "
              f"(n={sim_r.n_sims:,}  |  no API)")

        for label, prob, dec_odds in [
            ("Home Win", sim_r.p_home, home_odds),
            ("Draw",     sim_r.p_draw, draw_odds),
            ("Away Win", sim_r.p_away, away_odds),
        ]:
            if dec_odds <= 1.0:
                continue
            implied = 1.0 / dec_odds
            edge    = prob - implied
            ev      = prob * dec_odds - 1.0
            net     = dec_odds - 1.0
            ds      = decision_score(prob, edge, ev)
            kf_full = max(ev / net, 0.0) if net > 0 else 0.0
            stake   = round(kf_full * _KF * bankroll, 1)

            passes  = ev >= ev_min and ds >= ds_min
            marker  = "🔥" if passes else "  "
            print(f"   {marker} {label:<10} | odds={dec_odds:.2f} | sim={prob:.1%} mkt={implied:.1%} "
                  f"edge={edge:+.1%} EV={ev:+.1%} DS={ds:.0f} → stake={stake:.1f} NIS")

            if passes:
                sniper_bets.append({
                    "match":    match_label,
                    "outcome":  label,
                    "odds":     dec_odds,
                    "sim_prob": prob,
                    "implied":  implied,
                    "edge":     edge,
                    "ev":       ev,
                    "ds":       ds,
                    "stake":    stake,
                })

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    if skipped:
        print(f"  ⏭  SKIPPED (no odds): {', '.join(skipped)}")

    if not sniper_bets:
        print(f"  ❌  NO SNIPER BETS — no bet clears EV≥{ev_min:.0%} + DS≥{ds_min:.0f}")
    else:
        total_stake = sum(b["stake"] for b in sniper_bets)
        print(f"  🎯  SNIPER BETS: {len(sniper_bets)} found  |  total stake: {total_stake:.1f} NIS")
        print("-" * 65)
        for b in sniper_bets:
            print(f"  ✅  {b['match']}  →  {b['outcome']}")
            print(f"      Odds {b['odds']:.2f}  |  Model {b['sim_prob']:.1%}  vs Market {b['implied']:.1%}")
            print(f"      Edge {b['edge']:+.1%}  |  EV {b['ev']:+.1%}  |  DS {b['ds']:.0f}")
            print(f"      💰  Stake: {b['stake']:.1f} NIS  (¼-Kelly × {bankroll:.0f} NIS)")
            print()

    print("=" * 65)
    print("  ✅  Analysis complete — zero external API calls made")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sniper Protocol Profit Engine")
    parser.add_argument("--bankroll", type=float, default=_BANKROLL)
    parser.add_argument("--ev-min",   type=float, default=_EV_MIN)
    parser.add_argument("--ds-min",   type=float, default=_DS_MIN)
    args = parser.parse_args()
    run(bankroll=args.bankroll, ev_min=args.ev_min, ds_min=args.ds_min)
