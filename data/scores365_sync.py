"""
scores365_sync.py — Fetch live standings from your 365Scores predictor league.

Uses your browser session cookie (no password stored, no login bot).
Set the cookie as GitHub Secret: SCORE365_AUTH_COOKIE

HOW TO GET THE COOKIE + URL (one-time setup, ~2 minutes):
  1. Open Chrome/Firefox, log in to 365scores.com
  2. Navigate to your predictor league standings page
  3. Open DevTools (F12) → Network tab → filter by "Fetch/XHR"
  4. Refresh the page
  5. Find the API call that returns the standings JSON
     (look for requests containing "predictor", "standings", "leaderboard", or "participants")
  6. Right-click the request → Copy → Copy as cURL
  7. From the cURL output extract:
       - The full URL  → paste into SCORES365_API_URL below
       - The Cookie header value → save as GitHub Secret SCORE365_AUTH_COOKIE
  8. Also find your name and leader's name exactly as they appear in the JSON

CONFIGURE THE THREE CONSTANTS BELOW, THEN COMMIT.
"""
from __future__ import annotations

import os
from typing import Optional

import requests

# ============================================================
# CONFIGURE THESE after inspecting your browser's network tab
# ============================================================
SCORES365_API_URL   = "PASTE_URL_HERE"
MY_PARTICIPANT_NAME = "PASTE_YOUR_NAME_AS_IT_APPEARS_IN_JSON"
LEADER_NAME         = "doron gadesh"
# ============================================================

# JSON path config — adjust if the response structure differs
# The parser walks: response[PARTICIPANTS_ROOT_KEY] (list of participant objects)
# and looks for each participant's name at obj[NAME_KEY] and points at obj[POINTS_KEY].
PARTICIPANTS_ROOT_KEY = "participants"   # top-level key holding the list
NAME_KEY              = "name"          # key for participant's display name
POINTS_KEY            = "points"        # key for participant's total points


def _config_is_placeholder() -> bool:
    return SCORES365_API_URL.startswith("PASTE_") or MY_PARTICIPANT_NAME.startswith("PASTE_")


def fetch_standings(auth_cookie: Optional[str] = None) -> Optional[dict]:
    """
    Fetch live standings from 365Scores predictor league.

    Returns:
      {"my_points": int, "leader_points": int, "leader_name": str}
    or None on any failure (caller falls back to tournament_state.py hardcoded values).

    auth_cookie: raw Cookie header string (e.g. "session=abc; user_id=123")
    Falls back to env var SCORE365_AUTH_COOKIE if not provided.
    """
    auth_cookie = auth_cookie or os.environ.get("SCORE365_AUTH_COOKIE", "")

    if _config_is_placeholder():
        print(
            "[standings] scores365_sync.py is not configured yet.\n"
            "  → Open 365scores.com in your browser, inspect the Network tab,\n"
            "    and paste the API URL + your name into data/scores365_sync.py.\n"
            "  → Falling back to hardcoded values from tournament_state.py."
        )
        return None

    if not auth_cookie:
        print("[standings] SCORE365_AUTH_COOKIE env var not set — using hardcoded fallback.")
        return None

    headers = {
        "Cookie":     auth_cookie,
        "User-Agent": "Mozilla/5.0 (compatible; MondialPredictor/1.0)",
        "Accept":     "application/json",
    }

    try:
        resp = requests.get(SCORES365_API_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 401:
            print("[standings] 401 Unauthorized — session cookie has expired. "
                  "Update SCORE365_AUTH_COOKIE secret with a fresh cookie.")
        else:
            print(f"[standings] HTTP {status} from 365Scores API: {exc}")
        return None
    except Exception as exc:
        print(f"[standings] Failed to fetch standings: {exc}")
        return None

    # Parse the participant list
    participants = data
    for key in PARTICIPANTS_ROOT_KEY.split("."):
        if isinstance(participants, dict):
            participants = participants.get(key)
        else:
            break

    if not isinstance(participants, list):
        print(f"[standings] Could not find participant list at key '{PARTICIPANTS_ROOT_KEY}'. "
              f"Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        return None

    my_points:     Optional[int] = None
    leader_points: Optional[int] = None
    leader_name:   str           = LEADER_NAME

    for participant in participants:
        name   = str(participant.get(NAME_KEY, "")).strip().lower()
        points = participant.get(POINTS_KEY)

        if name == MY_PARTICIPANT_NAME.lower():
            try:
                my_points = int(points)
            except (TypeError, ValueError):
                pass

        if name == LEADER_NAME.lower():
            try:
                leader_points = int(points)
            except (TypeError, ValueError):
                pass

    if my_points is None or leader_points is None:
        # Try to find the actual leader (highest points) as fallback
        try:
            best = max(participants, key=lambda p: int(p.get(POINTS_KEY, 0)))
            if leader_points is None:
                leader_points = int(best.get(POINTS_KEY, 0))
                leader_name   = str(best.get(NAME_KEY, LEADER_NAME))
        except (ValueError, TypeError):
            pass

    if my_points is None:
        print(f"[standings] Could not find '{MY_PARTICIPANT_NAME}' in participants. "
              f"Check MY_PARTICIPANT_NAME in scores365_sync.py.")
        return None

    if leader_points is None:
        print("[standings] Could not determine leader points. Using hardcoded fallback.")
        return None

    print(f"[standings] Live: me={my_points}pts | leader={leader_points}pts ({leader_name})")
    return {
        "my_points":     my_points,
        "leader_points": leader_points,
        "leader_name":   leader_name,
    }
