"""
app.py — Mondial Predictor 2026 Dashboard

Streamlit dashboard for visualizing prediction history and today's picks.
Reads data/history.json and data/morning_picks.json directly from the repo.

Deploy: Streamlit Community Cloud → share.streamlit.io
  New app → repo: this repo → branch: main → main file: app.py
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mundial Predictor 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Layout ──────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* ── KPI cards ───────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1d2e 0%, #1e2236 100%);
    border-radius: 14px;
    padding: 20px 22px;
    border-left: 4px solid #00b4d8;
    box-shadow: 0 4px 18px rgba(0,180,216,0.12);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,180,216,0.22);
}
[data-testid="metric-container"] label {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: #7a8499 !important;
}
[data-testid="stMetricValue"] {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #eef2ff !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.80rem !important;
    color: #5a6478 !important;
}

/* ── Typography ──────────────────────────────────────── */
h1 { font-size: 2.0rem !important; font-weight: 800 !important; letter-spacing: -0.02em !important; }
h2 { font-size: 1.25rem !important; font-weight: 700 !important; }
p  { font-size: 0.95rem !important; line-height: 1.6 !important; }

/* ── Tables ──────────────────────────────────────────── */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }
[data-testid="stDataFrameResizable"] th {
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #7a8499 !important;
    text-align: center !important;
}
[data-testid="stDataFrameResizable"] td {
    font-size: 20px !important;   /* 20px renders flag emoji crisply */
    text-align: center !important;
}

/* ── Live badge ──────────────────────────────────────── */
.live-badge {
    display: inline-block;
    background: rgba(46,196,182,0.15);
    color: #2ec4b6;
    border: 1px solid rgba(46,196,182,0.35);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    vertical-align: middle;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] { background: #10121f !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li { font-size: 0.88rem !important; }

/* ── Footer ──────────────────────────────────────────── */
.footer-wrap {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-top: 1px solid #252840;
    color: #40485a;
    font-size: 0.80rem;
    line-height: 2.2;
}
.footer-wrap a { color: #00b4d8 !important; text-decoration: none !important; }
.footer-wrap a:hover { text-decoration: underline !important; }

/* ── Mobile ──────────────────────────────────────────── */
@media (max-width: 768px) {
    .main .block-container { padding-left: 0.6rem !important; padding-right: 0.6rem !important; }
    h1 { font-size: 1.55rem !important; }
    h2 { font-size: 1.05rem !important; }
    [data-testid="stMetricValue"]    { font-size: 1.6rem !important; }
    [data-testid="metric-container"] { padding: 13px 15px !important; }
    [data-testid="stMetricDelta"]    { display: none !important; }
}

/* ── Hide Streamlit chrome ───────────────────────────── */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
HISTORY_PATH = ROOT / "data" / "history.json"
PICKS_PATH   = ROOT / "data" / "morning_picks.json"

STAGE_EN: dict[str, str] = {
    "שלב הבתים":   "Group Stage",
    "32 האחרונות": "Round of 32",
    "שמינית גמר":  "Round of 16",
    "רבע גמר":     "Quarter-Final",
    "חצי גמר":     "Semi-Final",
    "מקום שלישי":  "3rd Place",
    "הגמר הגדול":  "Final",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#adb5bd", size=12),
    margin=dict(l=0, r=0, t=10, b=0),
)

# ── Country flag lookup ───────────────────────────────────────────────────────
_FLAGS: dict[str, str] = {
    # CONMEBOL
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "Uruguay": "🇺🇾", "Colombia": "🇨🇴",
    "Ecuador": "🇪🇨", "Venezuela": "🇻🇪", "Paraguay": "🇵🇾", "Chile": "🇨🇱",
    "Peru": "🇵🇪", "Bolivia": "🇧🇴",
    # UEFA
    "France": "🇫🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Germany": "🇩🇪", "Spain": "🇪🇸",
    "Netherlands": "🇳🇱", "Portugal": "🇵🇹", "Italy": "🇮🇹", "Belgium": "🇧🇪",
    "Croatia": "🇭🇷", "Denmark": "🇩🇰", "Austria": "🇦🇹", "Switzerland": "🇨🇭",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Turkey": "🇹🇷", "Serbia": "🇷🇸",
    "Czech Republic": "🇨🇿", "Hungary": "🇭🇺", "Slovakia": "🇸🇰",
    "Romania": "🇷🇴", "Georgia": "🇬🇪", "Slovenia": "🇸🇮", "Ukraine": "🇺🇦",
    "Albania": "🇦🇱", "Poland": "🇵🇱", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿", "Greece": "🇬🇷",
    "Norway": "🇳🇴", "Sweden": "🇸🇪",
    # CONCACAF
    "USA": "🇺🇸", "United States": "🇺🇸", "Mexico": "🇲🇽", "Canada": "🇨🇦",
    "Jamaica": "🇯🇲", "Honduras": "🇭🇳", "Costa Rica": "🇨🇷", "Panama": "🇵🇦",
    "Trinidad & Tobago": "🇹🇹", "El Salvador": "🇸🇻", "Guatemala": "🇬🇹",
    # CAF
    "Morocco": "🇲🇦", "Senegal": "🇸🇳", "Nigeria": "🇳🇬", "Egypt": "🇪🇬",
    "Cameroon": "🇨🇲", "South Africa": "🇿🇦", "Ghana": "🇬🇭", "Tunisia": "🇹🇳",
    "Ivory Coast": "🇨🇮", "Côte d'Ivoire": "🇨🇮", "Algeria": "🇩🇿",
    "Mali": "🇲🇱", "DR Congo": "🇨🇩", "Angola": "🇦🇴", "Zambia": "🇿🇲",
    "Cape Verde Islands": "🇨🇻", "Cape Verde": "🇨🇻", "Mozambique": "🇲🇿",
    # AFC
    "Japan": "🇯🇵", "South Korea": "🇰🇷", "Australia": "🇦🇺", "Iran": "🇮🇷",
    "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦", "Jordan": "🇯🇴", "UAE": "🇦🇪",
    "Uzbekistan": "🇺🇿", "Indonesia": "🇮🇩", "Iraq": "🇮🇶", "Oman": "🇴🇲",
    "China": "🇨🇳", "Bahrain": "🇧🇭", "Palestine": "🇵🇸", "India": "🇮🇳",
    # OFC
    "New Zealand": "🇳🇿",
}


def get_flag(team_name: str) -> str:
    """
    Return team name with flag prefix.
    - If the name already starts with a non-ASCII char (flag already present), return as-is.
    - Otherwise try exact match, then substring match.
    - Falls back to the original name with no flag if nothing matches.
    """
    if not team_name:
        return team_name
    # Already has a flag (first char is a non-ASCII emoji)
    if not team_name[0].isascii():
        return team_name
    clean = team_name.strip()
    if clean in _FLAGS:
        return f"{_FLAGS[clean]} {clean}"
    # Substring match — handles "IR Iran" → matches "Iran"
    for key, flag in _FLAGS.items():
        if key.lower() in clean.lower():
            return f"{flag} {clean}"
    return clean


def flag_short(team_name: str) -> str:
    """Flag + first word only — compact label for chart axes."""
    flagged = get_flag(team_name)
    parts   = flagged.split()
    if not parts:
        return team_name
    # Non-ASCII first token = emoji → return flag + next word
    if not parts[0].isascii() and len(parts) > 1:
        return f"{parts[0]} {parts[1]}"
    return parts[0]


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_history() -> pd.DataFrame:
    """Load and parse data/history.json."""
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        records = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df["stage_en"] = df["stage"].map(STAGE_EN).fillna(df["stage"])
        for col in ("predicted_home", "predicted_away", "actual_home", "actual_away",
                    "points_earned", "points_possible"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        st.warning(f"Could not parse history.json: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_picks() -> pd.DataFrame:
    """Load and parse data/morning_picks.json."""
    if not PICKS_PATH.exists():
        return pd.DataFrame()
    try:
        records = json.loads(PICKS_PATH.read_text(encoding="utf-8"))
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if "stage" in df.columns:
            df["stage_en"] = df["stage"].map(STAGE_EN).fillna(df["stage"])
        for col in ("final_home_goals", "final_away_goals"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        for col in ("lambda_home", "lambda_away"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
        return df
    except Exception as exc:
        st.warning(f"Could not parse morning_picks.json: {exc}")
        return pd.DataFrame()


# ── Helpers ──────────────────────────────────────────────────────────────────
def fmt_score(h: int, a: int) -> str:
    return f"{h} – {a}"


def result_badge(row: pd.Series) -> str:
    if row.get("exact_match"):
        return "🎯 Exact"
    if row.get("correct_result"):
        return "✅ Correct"
    return "❌ Wrong"


_HIGH_EV_THRESHOLD = 0.20   # matches ensemble.py's _HIGH_VALUE_THRESHOLD


def ev_badge(row: pd.Series) -> str:
    """
    Color-coded EV badge.
    ⭐ gold  — EV ≥ +20% (high-confidence value bet, ensemble gets value-priority prompt)
    🟢 green — EV > 0 but < +20% (standard value bet)
    🔴 red   — EV ≤ 0 (bookmaker has the edge)
    —        — no odds loaded for this match
    """
    ev      = row.get("ev_winner")
    outcome = row.get("ev_winner_outcome")
    if ev is None or (isinstance(ev, float) and pd.isna(ev)):
        return "—"
    lbl = str(outcome).title() if outcome else ""
    if ev >= _HIGH_EV_THRESHOLD:
        return f"⭐ {ev:+.1%} {lbl}"
    return f"🟢 {ev:+.1%} {lbl}" if ev > 0 else f"🔴 {ev:+.1%} {lbl}"


def plotly_chart(fig: go.Figure, height: int = 300) -> None:
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Load data ────────────────────────────────────────────────────────────────
hist  = load_history()
picks = load_picks()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Mundial 2026")
    st.markdown("**AI-Powered Match Predictor**")
    st.divider()

    st.markdown("#### 🧠 How It Works")
    st.markdown(
        "- **Poisson model** calibrated from live bookmaker odds\n"
        "- **Monte Carlo** simulation (10k draws per match)\n"
        "- **Claude AI** (Anthropic) calibrates using injury reports, form & lineup data\n"
        "- **Kelly criterion** identifies value bets"
    )
    st.divider()

    n_hist = len(hist)
    if n_hist:
        n_correct = int(hist["correct_result"].sum())
        pct_sb    = n_correct / n_hist
        st.markdown("#### 📊 Season Stats")
        st.markdown(f"**{n_correct}/{n_hist}** correct results")
        st.progress(pct_sb, text=f"{pct_sb*100:.0f}% accuracy")

    st.divider()

    st.markdown("#### 📬 Stay Updated")
    st.markdown(
        "Want daily picks sent to your WhatsApp before every match?\n\n"
        "📩 **[Contact / Subscribe](#)** *(coming soon)*"
    )
    st.caption("Daily picks at 09:00 IDT · Free during WC 2026")


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# ⚽ Mundial Predictor 2026")
st.markdown(
    f'<span class="live-badge">🔴 Live</span>'
    f'&nbsp;&nbsp;Last updated: <strong>{datetime.utcnow().strftime("%d %b %Y · %H:%M UTC")}</strong>',
    unsafe_allow_html=True,
)
st.caption("Poisson · Monte Carlo · Claude AI · Kelly Criterion")
st.divider()

# ── KPI Metrics ──────────────────────────────────────────────────────────────
total    = len(hist)
correct  = int(hist["correct_result"].sum())  if total else 0
exact    = int(hist["exact_match"].sum())     if total else 0
pts_earn = int(hist["points_earned"].sum())   if total else 0
pts_poss = int(hist["points_possible"].sum()) if total else 0

pct_correct = f"{correct/total*100:.0f}%" if total else "—"
pct_exact   = f"{exact/total*100:.0f}%"   if total else "—"
pct_pts     = f"{pts_earn/pts_poss*100:.0f}% of {pts_poss}" if pts_poss else "—"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Predictions",      total,         help="Total matches scored so far")
col2.metric("Correct Results",  pct_correct,   f"{correct}/{total} matches",  help="Right 1X2 outcome")
col3.metric("Exact Score Hits", pct_exact,     f"{exact}/{total} matches",    help="Perfect scoreline")
col4.metric("Points Earned",    str(pts_earn), pct_pts,                       help="vs. maximum possible")

st.divider()

# ── Today's Predictions ──────────────────────────────────────────────────────
st.subheader("📋 Today's Predictions")
st.caption(
    "Scores are FDR-adjusted (vice-captain.com) and AI-calibrated (Claude claude-opus-4-6). "
    "λ = Poisson rate (expected goals per 90 min)."
)

if picks.empty:
    st.info("No picks found. Predictions are generated each morning by the GitHub Actions pipeline.")
else:
    picks_date = picks["date"].iloc[0].date() if "date" in picks.columns else None
    if picks_date and picks_date != date.today():
        st.warning(f"⚠️ Showing predictions from {picks_date.strftime('%d %b %Y')} — today's pipeline hasn't run yet.")

    display = picks.copy()
    display["Home Team"]       = display["home_team"].apply(get_flag)
    display["Away Team"]       = display["away_team"].apply(get_flag)
    display["Predicted Score"] = display.apply(
        lambda r: fmt_score(r["final_home_goals"], r["final_away_goals"]), axis=1
    )
    display["λ Home"] = display["lambda_home"]
    display["λ Away"] = display["lambda_away"]
    display["Stage"]  = display.get("stage_en", display.get("stage", ""))
    if "ev_winner" in display.columns:
        display["EV Bet"] = display.apply(ev_badge, axis=1)

    show_cols = ["Home Team", "Away Team", "Stage", "Predicted Score", "λ Home", "λ Away", "EV Bet"]
    show_cols = [c for c in show_cols if c in display.columns]

    st.dataframe(
        display[show_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "λ Home": st.column_config.NumberColumn("λ Home", format="%.2f"),
            "λ Away": st.column_config.NumberColumn("λ Away", format="%.2f"),
        },
    )

st.divider()

# ── Market vs AI Simulation ─────────────────────────────────────────────────
_SIM_COLS = {"sim_p_home", "sim_p_draw", "sim_p_away",
             "market_p_home", "market_p_draw", "market_p_away"}
has_sim = not picks.empty and _SIM_COLS.issubset(set(picks.columns))

if has_sim:
    st.subheader("🔥 Market vs AI Simulation")
    st.caption(
        "Bookmaker implied probability (after overround removal) vs "
        "Monte Carlo simulation (10,000 Poisson draws).  "
        "🔥 Edge > 5% = Value Bet · ⭐ Edge ≥ 20% = High-Confidence Value Bet "
        "(Claude's reasoning explicitly prioritises the statistical edge)"
    )

    sim_df = picks.copy()
    sim_df["Match"] = sim_df.apply(
        lambda r: f"{get_flag(r['home_team'])} vs {get_flag(r['away_team'])}", axis=1
    )

    fig_sim    = go.Figure()
    OUTCOMES   = [("Home", "p_home"), ("Draw", "p_draw"), ("Away", "p_away")]
    MKT_COLORS = ["#0077b6", "#4a4e69", "#774b3b"]
    SIM_COLORS = ["#00b4d8", "#9d8df1", "#f4a261"]

    for i, (label, key) in enumerate(OUTCOMES):
        fig_sim.add_trace(go.Bar(
            name=f"Market {label}",
            x=sim_df["Match"],
            y=(sim_df[f"market_{key}"] * 100).round(1),
            marker_color=MKT_COLORS[i],
            text=(sim_df[f"market_{key}"] * 100).round(1).astype(str) + "%",
            textposition="outside", textfont=dict(size=10),
        ))
        fig_sim.add_trace(go.Bar(
            name=f"Sim {label}",
            x=sim_df["Match"],
            y=(sim_df[f"sim_{key}"] * 100).round(1),
            marker_color=SIM_COLORS[i],
            text=(sim_df[f"sim_{key}"] * 100).round(1).astype(str) + "%",
            textposition="outside", textfont=dict(size=10),
        ))

    fig_sim.update_layout(
        **PLOTLY_LAYOUT, height=360,
        barmode="group",
        bargap=0.20, bargroupgap=0.05,
        legend=dict(orientation="h", y=1.15, x=0),
        yaxis=dict(gridcolor="#2c2f3e", title="Probability (%)", range=[0, 90]),
        xaxis=dict(gridcolor="#2c2f3e", tickangle=-15),
    )
    st.plotly_chart(fig_sim, use_container_width=True, config={"displayModeBar": False})

    tbl = sim_df.copy()
    for label, scol, mcol in [
        ("H", "sim_p_home", "market_p_home"),
        ("D", "sim_p_draw", "market_p_draw"),
        ("A", "sim_p_away", "market_p_away"),
    ]:
        tbl[f"Mkt {label}"]  = (tbl[mcol] * 100).round(1).astype(str) + "%"
        tbl[f"Sim {label}"]  = (tbl[scol] * 100).round(1).astype(str) + "%"
        raw_edge = (tbl[scol] - tbl[mcol]) * 100
        tbl[f"Edge {label}"] = raw_edge.round(1).apply(lambda x: f"{x:+.1f}%")

    def _sim_value_badge(row: pd.Series) -> str:
        vb = row.get("sim_value_bet", "")
        if not vb or str(vb) in ("nan", "None", ""):
            return "—"
        edge_map = {
            "home": row.get("sim_p_home", 0) - row.get("market_p_home", 0),
            "draw": row.get("sim_p_draw", 0) - row.get("market_p_draw", 0),
            "away": row.get("sim_p_away", 0) - row.get("market_p_away", 0),
        }
        edge  = edge_map.get(str(vb).lower(), 0.0)
        label = str(vb).title()
        if edge >= _HIGH_EV_THRESHOLD:
            return f"⭐ {label} ({edge:+.0%})"
        return f"🔥 {label}"

    tbl["Value Bet"] = tbl.apply(_sim_value_badge, axis=1)

    show = ["Match",
            "Mkt H", "Sim H", "Edge H",
            "Mkt D", "Sim D", "Edge D",
            "Mkt A", "Sim A", "Edge A",
            "Value Bet"]
    st.dataframe(tbl[show], use_container_width=True, hide_index=True)
    st.divider()

# ── Charts (only when history exists) ────────────────────────────────────────
if hist.empty:
    st.info("📭 No historical data yet. Scores are logged automatically each morning after matches finish.")
else:
    # Row 1: Cumulative points | Accuracy donut
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("📈 Cumulative Points")
        cum = hist.copy()
        cum["Earned"]   = cum["points_earned"].cumsum()
        cum["Possible"] = cum["points_possible"].cumsum()
        cum["label"]    = cum["date"].dt.strftime("%-d %b")

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cum["label"], y=cum["Earned"],
            name="Points Earned", mode="lines+markers",
            line=dict(color="#00b4d8", width=2.5), marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(0,180,216,0.08)",
        ))
        fig_cum.add_trace(go.Scatter(
            x=cum["label"], y=cum["Possible"],
            name="Max Possible", mode="lines",
            line=dict(color="#4a4e69", width=1.5, dash="dot"),
        ))
        fig_cum.update_layout(
            **PLOTLY_LAYOUT, height=280,
            legend=dict(orientation="h", y=1.12, x=0),
            yaxis=dict(gridcolor="#2c2f3e", zeroline=False),
            xaxis=dict(gridcolor="#2c2f3e"),
        )
        st.plotly_chart(fig_cum, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.subheader("🎯 Accuracy Breakdown")
        exact_n   = exact
        correct_n = correct - exact
        wrong_n   = total - correct

        fig_donut = go.Figure(go.Pie(
            labels=["Exact Score", "Correct Result", "Wrong"],
            values=[exact_n, correct_n, wrong_n],
            hole=0.62,
            marker_colors=["#2ec4b6", "#ffd166", "#e63946"],
            textfont=dict(size=12),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{pct_correct}</b><br><span style='font-size:11px'>hit rate</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#f8f9fa"),
        )
        fig_donut.update_layout(
            **PLOTLY_LAYOUT, height=280,
            showlegend=True,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # Row 2: Strategy vs Result bar chart (last 10 matches)
    st.subheader("🔄 Strategy vs Result — Last 10 Matches")

    recent = hist.tail(10).copy()
    recent["Match"] = recent.apply(
        lambda r: f"{flag_short(r['home_team'])} vs {flag_short(r['away_team'])}", axis=1
    )

    fig_bar = go.Figure()
    for label, col, color in [
        ("Pred. Home", "predicted_home", "#00b4d8"),
        ("Actual Home", "actual_home",   "#0077b6"),
        ("Pred. Away", "predicted_away", "#ffd166"),
        ("Actual Away", "actual_away",   "#f4a261"),
    ]:
        fig_bar.add_trace(go.Bar(
            name=label, x=recent["Match"], y=recent[col],
            marker_color=color, text=recent[col],
            textposition="outside", textfont=dict(size=11),
        ))

    fig_bar.update_layout(
        **PLOTLY_LAYOUT, height=320,
        barmode="group",
        bargap=0.25, bargroupgap=0.06,
        legend=dict(orientation="h", y=1.12, x=0),
        yaxis=dict(gridcolor="#2c2f3e", dtick=1, title="Goals"),
        xaxis=dict(gridcolor="#2c2f3e", tickangle=-25),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # Row 3: Points by Stage (only if multiple stages have been played)
    if hist["stage_en"].nunique() > 1:
        st.subheader("🏆 Points by Tournament Stage")

        stage_df = (
            hist.groupby("stage_en")[["points_earned", "points_possible"]]
            .sum()
            .reset_index()
            .sort_values("points_possible", ascending=False)
        )

        fig_stage = go.Figure()
        fig_stage.add_trace(go.Bar(
            name="Max Possible", x=stage_df["stage_en"], y=stage_df["points_possible"],
            marker_color="#2c2f3e",
        ))
        fig_stage.add_trace(go.Bar(
            name="Earned", x=stage_df["stage_en"], y=stage_df["points_earned"],
            marker_color="#00b4d8",
        ))
        fig_stage.update_layout(
            **PLOTLY_LAYOUT, height=260,
            barmode="overlay",
            legend=dict(orientation="h", y=1.12, x=0),
            yaxis=dict(gridcolor="#2c2f3e"),
            xaxis=dict(gridcolor="#2c2f3e"),
        )
        st.plotly_chart(fig_stage, use_container_width=True, config={"displayModeBar": False})

        st.divider()

    # Full history table (collapsible)
    with st.expander("📊 Full Prediction History", expanded=False):
        full = hist.copy()
        full["Date"]      = full["date"].dt.strftime("%-d %b %Y")
        full["Match"]     = full.apply(
            lambda r: f"{get_flag(r['home_team'])} vs {get_flag(r['away_team'])}", axis=1
        )
        full["Stage"]     = full["stage_en"]
        full["Predicted"] = full.apply(
            lambda r: fmt_score(r["predicted_home"], r["predicted_away"]), axis=1
        )
        full["Actual"] = full.apply(
            lambda r: fmt_score(r["actual_home"], r["actual_away"]), axis=1
        )
        full["Result"] = full.apply(result_badge, axis=1)
        full["Pts"]    = (
            full["points_earned"].astype(str) + " / " + full["points_possible"].astype(str)
        )

        st.dataframe(
            full[["Date", "Match", "Stage", "Predicted", "Actual", "Result", "Pts"]],
            use_container_width=True,
            hide_index=True,
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-wrap">
    <p>
        ⚽ <strong>Mundial Predictor 2026</strong>
        &nbsp;·&nbsp; AI by <a href="https://www.anthropic.com" target="_blank">Claude (Anthropic)</a>
        &nbsp;·&nbsp; Data: The Odds API · API-Football · vice-captain.com
    </p>
    <p>
        📬 Want daily picks on WhatsApp?
        &nbsp;<a href="mailto:placeholder@example.com">Contact us</a>
        &nbsp;·&nbsp; <a href="#">Subscribe</a> <em>(coming soon)</em>
    </p>
    <p style="color:#252840; font-size:0.72rem; margin-top:0.4rem;">
        For entertainment purposes only · Not financial or betting advice · Gamble responsibly
    </p>
</div>
""", unsafe_allow_html=True)
