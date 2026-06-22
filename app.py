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
    page_title="Mondial Predictor 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tighten default Streamlit padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* KPI card styling */
    [data-testid="metric-container"] {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #00b4d8;
    }
    [data-testid="metric-container"] label { color: #adb5bd !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        color: #f8f9fa !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #6c757d !important;
    }

    /* Table tweaks */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Hide default footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
HISTORY_PATH  = ROOT / "data" / "history.json"
PICKS_PATH    = ROOT / "data" / "morning_picks.json"

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
        # Ensure numeric columns
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


def plotly_chart(fig: go.Figure, height: int = 300) -> None:
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Load data ────────────────────────────────────────────────────────────────
hist  = load_history()
picks = load_picks()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("⚽ Mondial Predictor 2026")
st.caption(
    f"Live performance dashboard · "
    f"Updated: {datetime.utcnow().strftime('%d %b %Y %H:%M')} UTC"
)
st.divider()

# ── KPI Metrics ──────────────────────────────────────────────────────────────
total      = len(hist)
correct    = int(hist["correct_result"].sum())  if total else 0
exact      = int(hist["exact_match"].sum())     if total else 0
pts_earn   = int(hist["points_earned"].sum())   if total else 0
pts_poss   = int(hist["points_possible"].sum()) if total else 0

pct_correct = f"{correct/total*100:.0f}%" if total else "—"
pct_exact   = f"{exact/total*100:.0f}%"   if total else "—"
pct_pts     = f"{pts_earn/pts_poss*100:.0f}% of {pts_poss}" if pts_poss else "—"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Predictions",    total,         help="Matches scored so far")
col2.metric("Correct Results",      pct_correct,   f"{correct}/{total} matches",  help="Right 1X2 outcome")
col3.metric("Exact Score Hits",     pct_exact,     f"{exact}/{total} matches",    help="Perfect scoreline")
col4.metric("Points Earned",        str(pts_earn), pct_pts,                       help="vs. maximum possible")

st.divider()

# ── Today's Predictions ──────────────────────────────────────────────────────
st.subheader("📋 Today's Predictions")
st.caption("Scores are FDR-adjusted (via vice-captain.com) and AI-calibrated (Claude claude-opus-4-6). λ = Poisson rate (expected goals).")

if picks.empty:
    st.info("No picks found. Predictions are generated each morning by the GitHub Actions pipeline.")
else:
    # Show most-recent picks and note the date
    picks_date = picks["date"].iloc[0].date() if "date" in picks.columns else None
    if picks_date:
        is_today = picks_date == date.today()
        date_note = "today" if is_today else picks_date.strftime("%d %b %Y")
        if not is_today:
            st.warning(f"⚠️ Showing predictions from {date_note} — today's pipeline hasn't run yet.")

    display = picks.copy()
    display["Predicted Score"] = display.apply(
        lambda r: fmt_score(r["final_home_goals"], r["final_away_goals"]), axis=1
    )
    display["λ Home"] = display["lambda_home"]
    display["λ Away"] = display["lambda_away"]
    display["Stage"]  = display.get("stage_en", display.get("stage", ""))

    col_map = {
        "home_team": "Home Team",
        "away_team": "Away Team",
    }
    show_cols = ["home_team", "away_team", "Stage", "Predicted Score", "λ Home", "λ Away"]
    show_cols = [c for c in show_cols if c in display.columns]

    st.dataframe(
        display[show_cols].rename(columns=col_map),
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
        "Monte Carlo simulation (10,000 Poisson draws). Edge > 5% = Value Bet 🔥"
    )

    sim_df = picks.copy()
    sim_df["Match"] = sim_df["home_team"] + " vs " + sim_df["away_team"]

    # Grouped bar chart
    fig_sim = go.Figure()
    OUTCOMES = [("Home", "p_home"), ("Draw", "p_draw"), ("Away", "p_away")]
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
        xaxis=dict(gridcolor="#2c2f3e"),
    )
    st.plotly_chart(fig_sim, use_container_width=True, config={"displayModeBar": False})

    # Detail table with edge columns
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

    tbl["Value Bet"] = tbl.get("sim_value_bet", pd.Series([None] * len(tbl))).map(
        lambda x: f"🔥 {str(x).title()}" if x and str(x) not in ("nan", "None", "") else "—"
    )

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
    recent["Match"] = (
        recent["home_team"].str.split().str[0]
        + " vs "
        + recent["away_team"].str.split().str[0]
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
        full["Match"]     = full["home_team"] + " vs " + full["away_team"]
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
st.markdown("---")
st.caption(
    "Mondial Predictor 2026 · "
    "Data: API-Football · The Odds API · vice-captain.com · "
    "AI: Claude claude-opus-4-6 (Anthropic)"
)
