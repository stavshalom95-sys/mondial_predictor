"""
Verification script — run from project root:
    python scripts/verify_winner_odds.py

Reads winner_odds.json + data/morning_picks.json, prints the mapping
and EV table. No files are written.
"""
import sys
import os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.winner_odds_loader import load_and_match

rows = load_and_match()

print()
print("=" * 72)
print(f"  {'MATCH':<32} {'EV Home':>9} {'EV Draw':>9} {'EV Away':>9}  Best")
print("=" * 72)

for r in rows:
    label = f"{r.home_team} vs {r.away_team}"
    label = label[:31]

    if not r.matched:
        print(f"  {label:<32}  ⚠️  no odds found")
        continue

    if r.ev_home is None:
        print(f"  {label:<32}  (no sim probs — run full pipeline first)")
        continue

    badge = "✅" if (r.ev_winner is not None and r.ev_winner > 0) else "❌"
    print(
        f"  {label:<32} {r.ev_home:>+8.2%} {r.ev_draw:>+8.2%} {r.ev_away:>+8.2%}"
        f"  {badge} {r.ev_winner_outcome}({r.ev_winner:+.2%})"
    )

print("=" * 72)
matched   = sum(1 for r in rows if r.matched)
has_ev    = sum(1 for r in rows if r.ev_winner is not None)
value_bets = sum(1 for r in rows if r.ev_winner is not None and r.ev_winner > 0)
print(f"  {matched}/{len(rows)} picks matched · {has_ev} with EV · {value_bets} value bet(s) (EV > 0)")
print()
