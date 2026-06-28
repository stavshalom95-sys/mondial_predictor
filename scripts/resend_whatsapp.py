"""
scripts/resend_whatsapp.py
--------------------------
Re-sends the last morning WhatsApp message without re-running the full pipeline.
Reads today's picks from data/morning_picks.json and rebuilds + sends the message.

Usage:
    GREEN_API_INSTANCE_ID=xxx \
    GREEN_API_TOKEN=yyy \
    WHATSAPP_RECIPIENT_PHONE=972XXXXXXXXX \
    python scripts/resend_whatsapp.py

Or to just print the message without sending:
    python scripts/resend_whatsapp.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import io

# Force UTF-8 on Windows (avoids cp1255 crash on emoji)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from notifications.notifier import send_whatsapp_message

_PICKS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "morning_picks.json")


def _check_green_api_instance() -> None:
    """
    Warns if the Green-API instance appears to be sleeping.
    Prints a link to the console so the user can re-authorize quickly.
    """
    instance_id = os.environ.get("GREEN_API_INSTANCE_ID", "")
    api_token   = os.environ.get("GREEN_API_TOKEN", "")
    if not instance_id or not api_token:
        return  # can't check without creds

    try:
        import requests
        url  = f"https://api.green-api.com/waInstance{instance_id}/getStateInstance/{api_token}"
        resp = requests.get(url, timeout=10)
        if resp.ok:
            state = resp.json().get("stateInstance", "unknown")
            print(f"[green-api] Instance state: {state}")
            if state != "authorized":
                print(
                    f"[green-api] WARNING: instance is '{state}' — WhatsApp send will fail.\n"
                    f"  Fix: open https://console.green-api.com/ → Instances → "
                    f"Instance {instance_id} → Scan QR / Authorize."
                )
        else:
            print(f"[green-api] State check HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as exc:
        print(f"[green-api] State check failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-send last WhatsApp morning pick message")
    parser.add_argument("--dry-run", action="store_true", help="Print message, do not send")
    parser.add_argument("--picks-path", default=_PICKS_PATH, help="Path to morning_picks.json")
    args = parser.parse_args()

    # Load picks
    try:
        with open(args.picks_path, encoding="utf-8") as f:
            picks = json.load(f)
    except FileNotFoundError:
        print(f"[resend] ERROR: {args.picks_path} not found. Run the pipeline first.")
        return 1
    except json.JSONDecodeError as e:
        print(f"[resend] ERROR: could not parse picks JSON: {e}")
        return 1

    if not picks:
        print("[resend] No picks in morning_picks.json — nothing to send.")
        return 0

    today = picks[0].get("date", "unknown")
    n     = len(picks)
    print(f"[resend] Loaded {n} pick(s) for {today} from {args.picks_path}")

    def _score_desc(h: str, a: str, sh: int, sa: int) -> str:
        """Match notifier._describe_score() — LTR-safe, team names explicit."""
        if sh > sa:
            return f"{h} wins {sh}-{sa}"
        elif sa > sh:
            return f"{a} wins {sa}-{sh}"
        else:
            return f"Draw {sh}-{sa}"

    lines = [
        f"⚽ *תחזית מונדיאל שטראוס — {today}*",
        "",
    ]

    for p in picks:
        h     = p.get("home_team", "?")
        a     = p.get("away_team", "?")
        sh    = p.get("final_home_goals")
        sa    = p.get("final_away_goals")
        p_h   = p.get("sim_p_home",   0)
        p_d   = p.get("sim_p_draw",   0)
        p_a   = p.get("sim_p_away",   0)
        lh    = p.get("lambda_home")
        la    = p.get("lambda_away")
        is_ko = p.get("is_knockout", False)
        stage = p.get("stage", "")

        lines.append(f"🔵 *{h} vs {a}*  [{stage}]")
        lines.append(f"   📊 Sim (10k): H={p_h:.0%}  D={p_d:.0%}  A={p_a:.0%}")

        # xG / model depth
        if lh is not None and la is not None:
            xg = lh + la
            lines.append(f"   🔬 xG: {h} λ={lh:.2f} / {a} λ={la:.2f} → {xg:.1f} exp. goals")

        # Pick line(s)
        if sh is not None and sa is not None:
            desc = _score_desc(h, a, sh, sa)
            if is_ko:
                lines.append(f"   🏆 *365Scores: {desc}* _(incl. extra time / penalties)_")
                lines.append(f"   🎰 *90-min bet: {desc}* _(if 90 min ends in draw → bet settles as draw)_")
            else:
                lines.append(f"   ⚽ *Final Prediction: {desc}*")

        lines.append("")

    lines.append("_נשלח אוטומטית ע\"י Mondial Predictor_")
    message = "\n".join(lines)

    if args.dry_run:
        print("\n--- Message (dry-run, not sending) ---")
        print(message)
        print("--- End ---")
        return 0

    # Check Green-API instance state before trying to send
    _check_green_api_instance()

    print("[resend] Attempting to send WhatsApp message...")
    ok = send_whatsapp_message(message)
    if ok:
        print("[resend] SUCCESS — message delivered to WhatsApp.")
        return 0
    else:
        print("[resend] FAILED — check credentials and Green-API instance state.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
