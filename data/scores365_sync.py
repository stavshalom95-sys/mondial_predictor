"""
scores365_sync.py — Fetch live standings from the 365Scores predictor league.

Endpoint is publicly accessible — no auth cookie required.
Response structure (verified against live API 2026-06-21):
  {
    "table": {
      "groupID": 15554,
      "membersCount": 10,
      "members": [
        {"name": "doron gadesh", "score": "33", "rank": "1", ...},
        ...
        {"name": "Stav Shalom",  "score": "22", "rank": "9", ...}
      ]
    },
    "ok": true,
    ...
  }

The leader is auto-detected as the member with rank "1" (no hardcoded name needed),
which makes the code resilient to leadership changes throughout the tournament.
"""
from __future__ import annotations

from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Production constants — verified against live API
# ---------------------------------------------------------------------------
SCORES365_API_URL   = "https://wcg-il.365scores.com/Groups/GetGroupTable?lang=2&groupID=15554"
MY_PARTICIPANT_NAME = "Stav Shalom"


def fetch_standings() -> Optional[dict]:
    """
    Fetch live standings from the 365Scores WC predictor group.

    Returns:
      {"my_points": int, "leader_points": int, "leader_name": str}
    or None on any failure (caller falls back to tournament_state.py hardcoded values).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MondialPredictor/1.0)",
        "Accept":     "application/json",
    }

    try:
        resp = requests.get(SCORES365_API_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        print(f"[standings] HTTP {status} fetching 365Scores: {exc}")
        return None
    except Exception as exc:
        print(f"[standings] Failed to fetch standings: {exc}")
        return None

    # Check API-level error flag
    if not data.get("ok", True):
        error = data.get("errorMessage") or data.get("error") or "unknown error"
        print(f"[standings] 365Scores API returned an error: {error}")
        return None

    # Navigate to the members list: data["table"]["members"]
    try:
        members = data["table"]["members"]
    except (KeyError, TypeError) as exc:
        print(f"[standings] Unexpected response shape — could not find table.members: {exc}")
        print(f"[standings] Top-level keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        return None

    if not isinstance(members, list) or len(members) == 0:
        print("[standings] members list is empty or missing.")
        return None

    my_points:     Optional[int] = None
    leader_points: Optional[int] = None
    leader_name:   str           = ""

    for member in members:
        name = str(member.get("name", "")).strip()
        raw_score = member.get("score", member.get("totalScore"))  # "score" is primary; fallback to "totalScore"
        raw_rank  = member.get("rank")

        try:
            points = int(raw_score)
        except (TypeError, ValueError):
            continue  # skip malformed entries

        # My points
        if name.lower() == MY_PARTICIPANT_NAME.lower():
            my_points = points

        # Leader = member with rank "1" (most robust — handles mid-tournament overtakes)
        if str(raw_rank) == "1":
            leader_points = points
            leader_name   = name

    # Fallback: if rank "1" wasn't found, pick the member with the highest score
    if leader_points is None:
        try:
            best = max(members, key=lambda m: int(m.get("score", m.get("totalScore", 0))))
            leader_points = int(best.get("score", best.get("totalScore", 0)))
            leader_name   = str(best.get("name", "")).strip()
        except (ValueError, TypeError):
            pass

    if my_points is None:
        print(
            f"[standings] Could not find '{MY_PARTICIPANT_NAME}' in the members list.\n"
            f"[standings] Names found: {[m.get('name') for m in members]}"
        )
        return None

    if leader_points is None:
        print("[standings] Could not determine leader. Check API response.")
        return None

    gap = leader_points - my_points
    print(f"[standings] Live: {MY_PARTICIPANT_NAME}={my_points}pts | "
          f"leader: {leader_name}={leader_points}pts | gap={gap}")

    return {
        "my_points":     my_points,
        "leader_points": leader_points,
        "leader_name":   leader_name,
    }
