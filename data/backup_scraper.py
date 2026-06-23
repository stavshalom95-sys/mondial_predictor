"""
data/backup_scraper.py — ESPN public-feed fallback for pre-match context.

No API key required.  ESPN's internal JSON endpoint that backs espn.com
returns WC form strings, W-D-L tournament records, and top scorers for
every active World Cup fixture.

Role in the context priority chain:
    P1 — RapidAPI        fetch_match_context()        injuries + form + lineups
    P2 — ESPN public     fetch_match_context_espn()   form string + WC record + scorer
    P3 — Internal        (main.py form block)          schedule-derived goal averages

P2 is called automatically when P1 returns no content (RAPIDAPI_KEY absent
or fixture not found).  P3 is always appended regardless of P1/P2 result.

What ESPN provides for WC 2026:
    • competitors[].form              — recent 5-match form across all competitions
                                        e.g. "DWWWD" (newest → oldest)
    • competitors[].records[].summary — WC-specific W-D-L e.g. "1-0-1"
    • competitors[].leaders[]         — top WC scorer with goal count

What ESPN does NOT provide (still needs RapidAPI):
    • Injuries / suspensions
    • Confirmed lineups
    • Head-to-head history

Public API:
    fetch_match_context_espn(home_team, away_team) -> Optional[ScraperContext]
    ScraperContext.to_prompt_section()              -> str (for Claude prompt)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
)
_TIMEOUT = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    return name.lower().strip()


def _teams_match(espn_name: str, schedule_name: str) -> bool:
    """True when ESPN displayName and schedule name refer to the same team."""
    a, b = _norm(espn_name), _norm(schedule_name)
    return a == b or a in b or b in a


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class ScraperContext:
    """Pre-match context extracted from ESPN's public WC scoreboard."""
    home_team:   str
    away_team:   str
    home_form:   str = ""   # recent form across all competitions, newest→oldest
    away_form:   str = ""
    home_record: str = ""   # WC record as "W-D-L" e.g. "1-0-1"
    away_record: str = ""
    home_scorer: str = ""   # e.g. "João Neves (1 G)"
    away_scorer: str = ""

    def to_prompt_section(self) -> str:
        """Format context for the Claude AI prompt."""
        lines = []
        for team, form, record, scorer in (
            (self.home_team, self.home_form, self.home_record, self.home_scorer),
            (self.away_team, self.away_form, self.away_record, self.away_scorer),
        ):
            if not form and not record:
                continue
            parts = []
            if record:
                parts.append(f"WC record (W-D-L): {record}")
            if form:
                parts.append(f"recent form (newest first): {form}")
            lines.append(f"{team} — {' | '.join(parts)}")
            if scorer:
                lines.append(f"  Top WC scorer: {scorer}")
        return "\n".join(lines)

    @property
    def has_content(self) -> bool:
        return bool(
            self.home_form or self.away_form
            or self.home_record or self.away_record
        )


# ── Private helper ────────────────────────────────────────────────────────────

def _extract_competitor(c: dict) -> tuple[str, str, str]:
    """Return (form, wc_record, top_scorer) from an ESPN competitor dict."""
    form   = c.get("form", "")
    recs   = c.get("records", [])
    record = recs[0].get("summary", "") if recs else ""
    scorer = ""
    for leader_group in c.get("leaders", []):
        candidates = leader_group.get("leaders", [])
        if candidates:
            top   = candidates[0]
            name  = top.get("athlete", {}).get("displayName", "")
            value = top.get("displayValue", "")
            unit  = leader_group.get("abbreviation", "")
            if name:
                scorer = f"{name} ({value} {unit})"
            break
    return form, record, scorer


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_match_context_espn(
    home_team:   str,
    away_team:   str,
    target_date: Optional[str] = None,
) -> Optional[ScraperContext]:
    """
    Fetch pre-match context from ESPN's public FIFA World Cup scoreboard.

    Args:
        home_team:   Schedule home team name (canonical form from football-data.org).
        away_team:   Schedule away team name.
        target_date: ISO date "YYYY-MM-DD" (defaults to today UTC).

    Returns:
        ScraperContext with form/record/scorer data, or None on any failure.
        Never raises — all errors are caught and logged.
    """
    if not _REQUESTS_AVAILABLE:
        print("[espn] 'requests' not installed — skipping ESPN fallback.")
        return None

    today_str = target_date or date.today().isoformat()
    dates_q   = today_str.replace("-", "")   # ESPN expects YYYYMMDD

    try:
        resp = _requests.get(
            _SCOREBOARD_URL,
            params={"dates": dates_q},
            timeout=_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MundialBot/1.0)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except _requests.exceptions.SSLError:
        # SSL cert verification fails on some Windows environments; retry without verify.
        # GitHub Actions (Ubuntu) does not have this issue.
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            resp = _requests.get(
                _SCOREBOARD_URL,
                params={"dates": dates_q},
                timeout=_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0 (compatible; MundialBot/1.0)"},
                verify=False,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc2:
            print(f"[espn] Scoreboard request failed (SSL fallback): {exc2}")
            return None
    except Exception as exc:
        print(f"[espn] Scoreboard request failed: {exc}")
        return None

    events = data.get("events", [])
    if not events:
        print(f"[espn] No events in ESPN scoreboard for {today_str}")
        return None

    for event in events:
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp        = comps[0]
        competitors = comp.get("competitors", [])

        espn_home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        espn_away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not espn_home or not espn_away:
            continue

        h_name = espn_home.get("team", {}).get("displayName", "")
        a_name = espn_away.get("team", {}).get("displayName", "")

        # Try normal order, then swapped (neutral-venue edge case)
        normal  = _teams_match(h_name, home_team) and _teams_match(a_name, away_team)
        swapped = _teams_match(h_name, away_team) and _teams_match(a_name, home_team)
        if not (normal or swapped):
            continue

        print(f"[espn] Matched: ESPN '{h_name} vs {a_name}' → schedule '{home_team} vs {away_team}'")

        # Respect schedule orientation (home = schedule home)
        if swapped:
            espn_home, espn_away = espn_away, espn_home

        h_form, h_record, h_scorer = _extract_competitor(espn_home)
        a_form, a_record, a_scorer = _extract_competitor(espn_away)

        ctx = ScraperContext(
            home_team   = home_team,
            away_team   = away_team,
            home_form   = h_form,
            away_form   = a_form,
            home_record = h_record,
            away_record = a_record,
            home_scorer = h_scorer,
            away_scorer = a_scorer,
        )
        if ctx.has_content:
            return ctx

        print(f"[espn] Match found but no form/record data yet — likely pre-tournament.")
        return None

    print(f"[espn] '{home_team} vs {away_team}' not found in ESPN scoreboard for {today_str}")
    return None
