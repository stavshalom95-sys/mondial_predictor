"""
scripts/profit_engine.py — Local betting analysis tool (Sniper Protocol deprecated)

Reads winner_odds.json, builds a strength model from history.json,
runs 20k Monte Carlo simulations, and prints EV / edge analysis to stdout.

No external APIs called. No WhatsApp output. Local diagnostic use only.

Usage:
  python scripts/profit_engine.py [--bankroll 100] [--output data/profit_report.json]
"""
from __future__ import annotations

import sys, json, os, argparse
from datetime import datetime, timezone
sys.stdout.reconfigure(encoding="utf-8")

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)

from core.strength_model import build_strength_model
from core.simulator import simulate

_BANKROLL = 100.0
_KF       = 0.25    # quarter-Kelly


def run(
    bankroll:    float = _BANKROLL,
    output_path: str   = "",
) -> None:
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

    print()
    print("=" * 65)
    print("  BETTING ANALYSIS — EV & EDGE BREAKDOWN")
    print("=" * 65)
    print(f"  Bankroll: {bankroll:.0f} NIS  |  ¼-Kelly sizing")
    print(f"  No external APIs called — internal model + history only")
    print("=" * 65)

    all_results: list[dict] = []
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

        print(f"\n{match_label}")
        print(f"   λ  home={lam_h:.2f}  away={lam_a:.2f}")
        print(f"   Sim  H={sim_r.p_home:.1%}  D={sim_r.p_draw:.1%}  A={sim_r.p_away:.1%}")

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
            kf_full = max(ev / net, 0.0) if net > 0 else 0.0
            stake   = round(kf_full * _KF * bankroll, 1)

            marker = "+" if ev > 0 and prob >= 0.50 else " "
            print(f"   [{marker}] {label:<10} | odds={dec_odds:.2f} | sim={prob:.1%} mkt={implied:.1%} "
                  f"edge={edge:+.1%} EV={ev:+.1%} → stake={stake:.1f} NIS")

            all_results.append({
                "match":   match_label,
                "outcome": label,
                "odds":    dec_odds,
                "prob":    prob,
                "implied": implied,
                "edge":    edge,
                "ev":      ev,
                "stake":   stake,
            })

    print()
    print("=" * 65)
    if skipped:
        print(f"  Skipped (no odds): {', '.join(skipped)}")
    print("  Analysis complete")
    print("=" * 65)

    if output_path:
        report = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bankroll":     bankroll,
            "wc_matches":   sm.n_matches if sm else 0,
            "results":      all_results,
            "skipped":      skipped,
        }
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  Report written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Betting EV analysis (local diagnostic)")
    parser.add_argument("--bankroll", type=float, default=_BANKROLL)
    parser.add_argument("--output",   type=str,   default="", help="Path to write JSON report")
    args = parser.parse_args()
    run(bankroll=args.bankroll, output_path=args.output)
