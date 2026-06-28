# Mondial Predictor

> **AI-driven football match prediction engine for tournament competition and value betting — FIFA World Cup 2026**

---

## Overview

Mondial Predictor is a fully automated forecasting pipeline that combines statistical modelling, Monte Carlo simulation, and large language model reasoning to generate daily match predictions for the FIFA World Cup 2026.

The system runs end-to-end on GitHub Actions with zero manual intervention: it fetches the live schedule, pulls bookmaker odds, calibrates a Dixon-Coles Poisson model, runs a 10,000-iteration simulation, applies tournament context (group standings, tiebreaker scenarios, rotation risk), queries Claude for contextual refinement, computes Kelly-criterion bet sizing, and delivers a structured prediction report via WhatsApp — all before the first morning kick-off.

---

## Technical Stack

### Data Science
| Component | Implementation |
|---|---|
| **Odds Conversion** | Multiplicative overround removal → true 1X2 + O/U probabilities |
| **Poisson Calibration** | Two-pass grid search (Δ0.1 → Δ0.02) + scipy Nelder-Mead polish; Dixon-Coles τ correction on low-score cells |
| **Monte Carlo Simulation** | 10,000 match iterations per fixture; score grid with exact-score, 1X2, O/U marginals |
| **Bayesian Priors** | WC 2026 historical priors per team (`wc_priors.json`); blended with in-tournament strength estimates as match data accumulates |
| **Tournament Context** | Motivation multipliers (λ ×0.85 rotation → ×1.10 must-win); FIFA 2026 7-step tiebreaker chain; 3rd-place bubble detection |
| **Performance Tracking** | Brier score, model-vs-market alpha, per-source win rates, Kelly ROI — all logged to `data/history.json` |

### AI Layer
| Component | Implementation |
|---|---|
| **Model** | `claude-opus-4-6` via Anthropic SDK |
| **Structured Output** | `instructor` library (Pydantic + auto-retry on schema mismatch); falls back to raw SDK JSON parsing |
| **Context Injection** | Live lineups, injuries, recent form, H2H, tournament motivation, tiebreaker scenario |
| **Value-Bet Directive** | When Monte Carlo edge ≥ 20%, prompt explicitly instructs Claude to justify the statistical mispricing |
| **Timeout** | 30-second hard timeout on all API calls (`httpx.Client(timeout=30.0)`) — pipeline never hangs |
| **Offline Fallback** | If `ANTHROPIC_API_KEY` is absent or API fails, WhatsApp message explicitly reports `AI layer offline — running on baseline Poisson` |

### Automation
| Component | Implementation |
|---|---|
| **CI/CD** | GitHub Actions (`daily_run.yml`) |
| **Schedule** | Cron 05:00, 05:30, 06:00 UTC + manual `workflow_dispatch` |
| **Idempotency** | `last_run.json` guard prevents double-processing on same calendar day |
| **Schedule Sync** | `scripts/sync_winner_odds.py` — 3-case logic: STALE (past match → remove), PRESERVED (filled odds → keep), ADDED (new fixture → placeholder) |
| **Tables Sync** | `scripts/sync_tables.py` — live group standings via football-data.org; fallback to schedule-derived computation |
| **Conflict Handling** | `git pull --rebase origin main` before every push; CI data files resolved to remote on conflict |

### Delivery
| Component | Implementation |
|---|---|
| **Channel** | WhatsApp via Green-API |
| **Pre-flight Check** | Instance state verified before send (`getStateInstance`); `::error::` annotation on non-authorized |
| **Retry Logic** | 3 attempts with 0s / 3s / 5s backoff |
| **Message Format** | Per-match: simulation probabilities, λ values, xG, dual-track KO prediction (365Scores vs 90-min market), value bets, Kelly ticket, tournament context, performance stats |
| **Resend Utility** | `scripts/resend_whatsapp.py` — lightweight resend from `morning_picks.json` without re-running the pipeline |

---

## Features

### Real-Time Prediction Workflow
- Fetches today's fixtures from football-data.org every morning
- Pulls live 1X2, Over/Under, and corners odds from The Odds API
- Calibrates a separate Poisson model per match using current bookmaker prices
- Runs 10k Monte Carlo iterations and computes full score distribution
- Applies team strength estimates derived from all completed WC 2026 results

### Intelligent Odds Ingestion & Stale-Data Cleanup
- `winner_odds.json` stores multi-market odds in a structured dict format
- `sync_winner_odds.py` runs daily with explicit 3-case decision logic:
  - **STALE**: match has started or is marked `final` → entry removed
  - **PRESERVED**: upcoming match with filled odds → never overwritten
  - **ADDED**: upcoming match with no odds → blank placeholder inserted
- Zero-odds placeholders are silently filtered from EV calculations to prevent spurious −100% EV signals

### Value Betting Engine
- **Expected Value** computed per outcome: `EV = p_model × odds − 1`
- **Kelly Criterion** stake sizing: `k = (p × b − (1−p)) / b`, applied at half-Kelly
- Markets covered: 1X2 winner, Goals 0–1 / 2–3 / 4+ brackets, Corners 0–8 / 9–11 / 12+
- Daily budget cap enforced (100 NIS default); stakes scaled within budget
- Value bets and full Kelly ticket included in WhatsApp output

### Knockout Stage Support
- Dual-track predictions: **365Scores** (includes extra time / penalties) vs **90-min market** (draw = draw)
- Competition pick uses `_competition_score_pick()`: modal score in normal mode; second-most-likely same-direction score when trailing by ≥8 pts (variance mode)
- Poisson simulation treats 90-min as the base; ET/penalty probability handled separately in competition framing

### Robust Error Handling
- All external API calls wrapped in try/except with typed exception handling (`AuthenticationError`, `RateLimitError`, generic fallback)
- Claude API: 30-second hard timeout on both instructor and raw SDK paths
- The Odds API: graceful fallback to prior-only model when no odds returned
- Schedule fetch: cached `sample_games.json` used if live fetch fails
- Group tables: computed from schedule JSON if live standings API unavailable
- Result ingestion: RapidAPI → schedule JSON fallback chain

---

## Project Structure

```
core/
  ai_ensemble.py          # Claude contextual calibration; instructor + raw SDK paths
  odds_converter.py       # Market odds → true probabilities (overround removal)
  poisson_engine.py       # Dixon-Coles calibration; score grid; Monte Carlo
  strategy_advisor.py     # Conservative / contrarian / variance-mode logic
  tiebreaker.py           # FIFA 2026 7-step tiebreaker chain; 3rd-place bubble ranking

config/
  scoring_rules.py        # TournamentStage enum; SCORING dict; stage multipliers

data/
  context_fetcher.py      # API-Football (RapidAPI): lineups, injuries, H2H, form
  data_pipeline.py        # Schedule parsing; get_todays_matches()
  motivation.py           # Group qualification statuses; λ multipliers; tiebreaker notes
  odds_fetcher.py         # The Odds API: live 1X2 + O/U; team name normalisation
  performance_tracker.py  # History ingestion; Brier score; alpha vs market; ROI
  results_fetcher.py      # RapidAPI match results for history ingestion
  winner_odds_loader.py   # winner_odds.json parser; EV enrichment; format migration

notifications/
  notifier.py             # format_daily_message(); send_whatsapp_message() via Green-API

scripts/
  fetch_schedule.py       # football-data.org schedule fetch → JSON
  resend_whatsapp.py      # Lightweight resend from morning_picks.json
  sync_tables.py          # Live group standings fetch + fallback computation
  sync_winner_odds.py     # Daily odds file sync with 3-case stale/preserve/add logic

data/ (runtime files)
  morning_picks.json      # Today's predictions (committed by CI after each run)
  history.json            # All-time prediction history with scores and probabilities
  group_tables.json       # Live WC 2026 group standings
  winner_odds.json        # Today's bookmaker odds (multi-market dict format)
  last_run.json           # Idempotency guard (timestamp of last successful run)
  wc_priors.json          # Per-team Bayesian priors from WC historical data

.github/workflows/
  daily_run.yml           # Main CI pipeline: cron + dispatch
  lineup_check.yml        # Manual: re-runs Claude when confirmed XIs change

main.py                   # Pipeline orchestrator
```

---

## Setup

### Required GitHub Secrets

| Secret | Source |
|---|---|
| `FOOTBALL_DATA_API_KEY` | [football-data.org](https://www.football-data.org/client/register) — free Tier 0 |
| `THE_ODDS_API_KEY` | [the-odds-api.com](https://the-odds-api.com) — free 500 requests/month |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `RAPIDAPI_KEY` | [API-Football on RapidAPI](https://rapidapi.com/api-sports/api/api-football) |
| `SCORE365_AUTH_COOKIE` | Browser network tab — competition standings cookie |
| `GREEN_API_INSTANCE_ID` | [green-api.com](https://green-api.com) — WhatsApp sandbox instance |
| `GREEN_API_TOKEN` | green-api.com — instance token |
| `WHATSAPP_RECIPIENT_PHONE` | Target phone in format `972XXXXXXXXX` |

### Local Run

```bash
# Fetch today's schedule
python scripts/fetch_schedule.py --output tests/sample_games.json

# Dry run — prints pipeline output, no WhatsApp
python main.py tests/sample_games.json --no-notify

# Force re-run (override today's idempotency guard)
python main.py tests/sample_games.json --force --no-notify

# Resend last WhatsApp without re-running the pipeline
python scripts/resend_whatsapp.py

# Sync winner_odds.json to today's schedule
python scripts/sync_winner_odds.py tests/sample_games.json winner_odds.json
```

---

## Workflow

```
05:00 UTC  GitHub Actions trigger
    │
    ├── sync_tables.py          → group_tables.json
    ├── fetch_schedule.py       → today's fixtures
    ├── sync_winner_odds.py     → winner_odds.json (stale/preserve/add)
    │
    └── main.py
         ├── Ingest yesterday's results → history.json
         ├── For each match:
         │    ├── Fetch odds (The Odds API)
         │    ├── Calibrate Poisson model (Dixon-Coles)
         │    ├── Apply tournament context (motivation, tiebreaker)
         │    ├── Run 10k Monte Carlo simulation
         │    ├── Compute EV + Kelly stakes (winner_odds.json)
         │    ├── Call Claude (30s timeout) → contextual pick
         │    └── Build DailyPick + morning_data record
         │
         ├── Format WhatsApp message
         ├── Send via Green-API (3 retries)
         └── Commit morning_picks.json, history.json → main
```

---

## Disclaimer

**Mondial Predictor is an automated statistical forecasting tool.** Predictions are generated by probabilistic models calibrated from publicly available bookmaker odds and do not constitute financial advice.

- All match outcome probabilities are estimates derived from mathematical models. Past performance does not guarantee future results.
- Value betting recommendations are based on detected discrepancies between model-implied probabilities and market-implied probabilities. Edge estimates carry uncertainty.
- **Betting involves risk of financial loss.** Never bet more than you can afford to lose. Set strict daily and session limits before placing any wagers.
- This project is for personal use in a private friends' tournament context. It is not affiliated with any bookmaker, sportsbook, or gambling operator.
- If you or someone you know has a problem with gambling, contact the national helpline in your country.
