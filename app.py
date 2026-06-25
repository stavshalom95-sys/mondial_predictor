"""
app.py — Mundial Predictor 2026 Dashboard
Redesigned with UI/UX Pro Max skill:
  Style  : OLED Dark Mode + Bento Grid (Data-Dense Dashboard)
  Fonts  : Inter (body) · JetBrains Mono (numbers)
  Colors : Deep navy bg · Cyan accent · Amber value · Teal success · Gold exact
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

# ── Design System CSS ─────────────────────────────────────────────────────────
# Skill: OLED Dark Mode + Bento Grid (Data-Dense)
# Palette: deep-navy bg · cyan #00C8FF · amber #F59E0B · teal #2EC4B6 · gold #FFD166
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
/* ── Design Tokens ──────────────────────────────────────────────────────── */
:root {
  --bg:            #08091C;
  --surface:       #0D1128;
  --surface-alt:   #111630;
  --surface-hover: #151B38;

  --text-1: #EEF2FF;
  --text-2: #8892B0;
  --text-3: #4A5374;

  --cyan:      #00C8FF;
  --cyan-dim:  rgba(0,200,255,.13);
  --cyan-glow: 0 0 28px rgba(0,200,255,.18);

  --amber:     #F59E0B;
  --amber-dim: rgba(245,158,11,.13);

  --teal:      #2EC4B6;
  --teal-dim:  rgba(46,196,182,.13);

  --gold:      #FFD166;
  --gold-dim:  rgba(255,209,102,.13);

  --red:       #E63946;
  --red-dim:   rgba(230,57,70,.13);

  --purple:    #9D8DF1;
  --purple-dim:rgba(157,141,241,.13);

  --border:      rgba(255,255,255,.06);
  --border-cyan: rgba(0,200,255,.22);

  --r-card:  16px;
  --r-badge: 20px;
  --gap:     1rem;

  --ease: cubic-bezier(.4,0,.2,1);
}

/* ── Global reset ───────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  background: var(--bg) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  color: var(--text-1) !important;
}

.main .block-container {
  padding-top: 0 !important;
  padding-bottom: 3rem;
  max-width: 1180px;
}

/* ── Hero section ───────────────────────────────────────────────────────── */
.hero {
  position: relative;
  padding: 2.6rem 2.4rem 2.2rem;
  margin: -1px -1px 0;
  background: linear-gradient(135deg, #08091C 0%, #0D1230 55%, #0F142A 100%);
  border-bottom: 1px solid var(--border);
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 700px 300px at 80% 50%, rgba(0,200,255,.05) 0%, transparent 70%),
    radial-gradient(ellipse 400px 200px at 15% 80%, rgba(157,141,241,.05) 0%, transparent 70%);
  pointer-events: none;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(230,57,70,.12);
  border: 1px solid rgba(230,57,70,.3);
  border-radius: var(--r-badge);
  padding: 3px 12px 3px 8px;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: #E63946;
  margin-bottom: 1.1rem;
}
.hero-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #E63946;
  animation: pulse-dot 1.8s ease infinite;
}
@keyframes pulse-dot {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:.4; transform:scale(.7); }
}
.hero-title {
  font-size: 2.4rem;
  font-weight: 900;
  letter-spacing: -.03em;
  line-height: 1.1;
  color: var(--text-1);
  margin: 0 0 .6rem;
}
.hero-title .accent {
  background: linear-gradient(90deg, var(--cyan) 0%, #7C68EE 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-stack {
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
  margin-bottom: .9rem;
}
.hero-tag {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 3px 10px;
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .06em;
  color: var(--text-2);
  text-transform: uppercase;
}
.hero-update {
  font-size: .78rem;
  color: var(--text-3);
  font-family: 'JetBrains Mono', monospace;
}

/* ── KPI bento grid ─────────────────────────────────────────────────────── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--gap);
  margin: 1.6rem 0;
}
@media (max-width: 900px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 560px) { .kpi-grid { grid-template-columns: 1fr; } }

.kpi-card {
  position: relative;
  background: var(--surface);
  border-radius: var(--r-card);
  border: 1px solid var(--border);
  padding: 1.25rem 1.4rem;
  overflow: hidden;
  transition: transform .22s var(--ease), box-shadow .22s var(--ease);
  cursor: default;
}
.kpi-card:hover {
  transform: translateY(-3px);
}
.kpi-card::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  border-radius: var(--r-card) var(--r-card) 0 0;
}
.kpi-card.c-cyan  { border-left: 3px solid var(--cyan);   }
.kpi-card.c-teal  { border-left: 3px solid var(--teal);   }
.kpi-card.c-gold  { border-left: 3px solid var(--gold);   }
.kpi-card.c-purple{ border-left: 3px solid var(--purple); }

.kpi-card.c-cyan:hover   { box-shadow: 0 8px 28px rgba(0,200,255,.14); }
.kpi-card.c-teal:hover   { box-shadow: 0 8px 28px rgba(46,196,182,.14); }
.kpi-card.c-gold:hover   { box-shadow: 0 8px 28px rgba(255,209,102,.14); }
.kpi-card.c-purple:hover { box-shadow: 0 8px 28px rgba(157,141,241,.14); }

.kpi-glow {
  position: absolute;
  top: -40px; right: -40px;
  width: 120px; height: 120px;
  border-radius: 50%;
  opacity: .08;
  filter: blur(28px);
}
.c-cyan   .kpi-glow { background: var(--cyan); }
.c-teal   .kpi-glow { background: var(--teal); }
.c-gold   .kpi-glow { background: var(--gold); }
.c-purple .kpi-glow { background: var(--purple); }

.kpi-label {
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--text-3);
  margin-bottom: .55rem;
}
.kpi-value {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -.03em;
  line-height: 1;
  font-family: 'JetBrains Mono', monospace;
  margin-bottom: .35rem;
}
.c-cyan   .kpi-value { color: var(--cyan); }
.c-teal   .kpi-value { color: var(--teal); }
.c-gold   .kpi-value { color: var(--gold); }
.c-purple .kpi-value { color: var(--purple); }

.kpi-sub {
  font-size: .75rem;
  color: var(--text-3);
  font-weight: 500;
}

/* ── Section header ─────────────────────────────────────────────────────── */
.section-header {
  display: flex;
  align-items: center;
  gap: .65rem;
  margin: 2rem 0 .75rem;
}
.section-header .icon {
  font-size: 1.1rem;
}
.section-header .title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text-1);
  letter-spacing: -.01em;
}
.section-header .line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border-cyan), transparent);
  margin-left: .5rem;
}
.section-caption {
  font-size: .78rem;
  color: var(--text-3);
  margin-bottom .75rem;
  line-height: 1.6;
}

/* ── Gradient divider ───────────────────────────────────────────────────── */
.div-grad {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-cyan), transparent);
  margin: 1.8rem 0;
  border: none;
}

/* ── Streamlit metric override ─────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--surface) !important;
  border-radius: var(--r-card) !important;
  padding: 1.1rem 1.3rem !important;
  border: 1px solid var(--border) !important;
  border-left: 3px solid var(--cyan) !important;
  box-shadow: none !important;
  transition: transform .22s var(--ease), box-shadow .22s var(--ease) !important;
}
[data-testid="metric-container"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 22px rgba(0,200,255,.12) !important;
}
[data-testid="metric-container"] label {
  font-size: .68rem !important;
  font-weight: 700 !important;
  letter-spacing: .11em !important;
  text-transform: uppercase !important;
  color: var(--text-3) !important;
  font-family: 'Inter', sans-serif !important;
}
[data-testid="stMetricValue"] {
  font-size: 2rem !important;
  font-weight: 800 !important;
  color: var(--cyan) !important;
  font-family: 'JetBrains Mono', monospace !important;
  letter-spacing: -.03em !important;
}
[data-testid="stMetricDelta"] {
  font-size: .75rem !important;
  color: var(--text-3) !important;
}

/* ── Typography ─────────────────────────────────────────────────────────── */
h1, h2, h3 {
  font-family: 'Inter', sans-serif !important;
  letter-spacing: -.02em !important;
}
h1 { font-size: 2rem !important; font-weight: 900 !important; }
h2 { font-size: 1.15rem !important; font-weight: 700 !important; }
p  { font-size: .94rem !important; line-height: 1.65 !important; }

/* ── Tables ─────────────────────────────────────────────────────────────── */
.stDataFrame                         { border-radius: var(--r-card) !important; overflow: hidden !important; border: 1px solid var(--border) !important; }
[data-testid="stDataFrameResizable"] { background: var(--surface) !important; }
[data-testid="stDataFrameResizable"] th {
  font-size: .7rem !important;
  font-weight: 700 !important;
  letter-spacing: .09em !important;
  text-transform: uppercase !important;
  color: var(--text-3) !important;
  background: var(--surface-alt) !important;
  text-align: center !important;
}
[data-testid="stDataFrameResizable"] td {
  font-size: .9rem !important;
  text-align: center !important;
  color: var(--text-1) !important;
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: #060817 !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
  font-size: .86rem !important;
  color: var(--text-2) !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
  color: var(--text-1) !important;
}

/* ── Progress bar ───────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
  background: var(--cyan) !important;
  border-radius: 4px !important;
}
[data-testid="stProgress"] > div {
  background: var(--surface-alt) !important;
  border-radius: 4px !important;
}

/* ── Info / warning boxes ───────────────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--r-card) !important;
  border: 1px solid var(--border) !important;
}

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border-radius: var(--r-card) !important;
  border: 1px solid var(--border) !important;
}

/* ── Footer ─────────────────────────────────────────────────────────────── */
.footer {
  margin-top: 3rem;
  padding: 2rem 0 1.5rem;
  border-top: 1px solid var(--border);
  text-align: center;
}
.footer-logo {
  font-size: 1rem;
  font-weight: 800;
  letter-spacing: -.02em;
  color: var(--text-1);
  margin-bottom: .9rem;
}
.footer-logo .accent { color: var(--cyan); }
.footer-links {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: .4rem 1.5rem;
  margin-bottom: 1rem;
}
.footer-links a {
  font-size: .78rem;
  color: var(--text-3) !important;
  text-decoration: none !important;
  transition: color .15s var(--ease);
}
.footer-links a:hover { color: var(--cyan) !important; }
.footer-legal {
  font-size: .68rem;
  color: var(--text-3);
  opacity: .55;
}
.footer-powered {
  display: inline-flex;
  align-items: center;
  gap: .35rem;
  font-size: .73rem;
  color: var(--text-3);
  margin-bottom: .6rem;
}

/* ── Match card chips ───────────────────────────────────────────────────── */
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .06em;
  text-transform: uppercase;
}
.badge-exact   { background: var(--gold-dim);   color: var(--gold);   border: 1px solid rgba(255,209,102,.25); }
.badge-correct { background: var(--teal-dim);   color: var(--teal);   border: 1px solid rgba(46,196,182,.25); }
.badge-wrong   { background: var(--red-dim);    color: var(--red);    border: 1px solid rgba(230,57,70,.25); }
.badge-ev-high { background: var(--gold-dim);   color: var(--gold);   border: 1px solid rgba(255,209,102,.25); }
.badge-ev-pos  { background: var(--teal-dim);   color: var(--teal);   border: 1px solid rgba(46,196,182,.25); }
.badge-ev-neg  { background: var(--red-dim);    color: var(--red);    border: 1px solid rgba(230,57,70,.25); }
.badge-live    { background: rgba(230,57,70,.12); color: var(--red);  border: 1px solid rgba(230,57,70,.3); }

/* ── Mobile ─────────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
  .hero { padding: 1.8rem 1.2rem 1.6rem; }
  .hero-title { font-size: 1.8rem; }
  .main .block-container { padding-left: .8rem !important; padding-right: .8rem !important; }
  [data-testid="stMetricValue"]    { font-size: 1.6rem !important; }
  [data-testid="metric-container"] { padding: 1rem !important; }
}

/* ── Hide Streamlit chrome ──────────────────────────────────────────────── */
footer           { visibility: hidden; }
#MainMenu        { visibility: hidden; }
.stDeployButton  { display: none; }
[data-testid="stToolbar"] { display: none; }
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

# Plotly layout — matches OLED dark palette
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#8892B0", size=12),
    margin=dict(l=0, r=0, t=10, b=0),
)

# Chart accent colors (aligned to design system)
CLR_CYAN   = "#00C8FF"
CLR_CYAN_D = "#0077A3"
CLR_AMBER  = "#F59E0B"
CLR_AMBER_D= "#B87500"
CLR_TEAL   = "#2EC4B6"
CLR_RED    = "#E63946"
CLR_GOLD   = "#FFD166"
CLR_PURPLE = "#9D8DF1"
CLR_GRID   = "#1A1F38"

# ── Country flag lookup ───────────────────────────────────────────────────────
_FLAGS: dict[str, str] = {
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "Uruguay": "🇺🇾", "Colombia": "🇨🇴",
    "Ecuador": "🇪🇨", "Venezuela": "🇻🇪", "Paraguay": "🇵🇾", "Chile": "🇨🇱",
    "Peru": "🇵🇪", "Bolivia": "🇧🇴",
    "France": "🇫🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Germany": "🇩🇪", "Spain": "🇪🇸",
    "Netherlands": "🇳🇱", "Portugal": "🇵🇹", "Italy": "🇮🇹", "Belgium": "🇧🇪",
    "Croatia": "🇭🇷", "Denmark": "🇩🇰", "Austria": "🇦🇹", "Switzerland": "🇨🇭",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Turkey": "🇹🇷", "Serbia": "🇷🇸",
    "Czech Republic": "🇨🇿", "Hungary": "🇭🇺", "Slovakia": "🇸🇰",
    "Romania": "🇷🇴", "Georgia": "🇬🇪", "Slovenia": "🇸🇮", "Ukraine": "🇺🇦",
    "Albania": "🇦🇱", "Poland": "🇵🇱", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿", "Greece": "🇬🇷",
    "Norway": "🇳🇴", "Sweden": "🇸🇪",
    "USA": "🇺🇸", "United States": "🇺🇸", "Mexico": "🇲🇽", "Canada": "🇨🇦",
    "Jamaica": "🇯🇲", "Honduras": "🇭🇳", "Costa Rica": "🇨🇷", "Panama": "🇵🇦",
    "Trinidad & Tobago": "🇹🇹", "El Salvador": "🇸🇻", "Guatemala": "🇬🇹",
    "Morocco": "🇲🇦", "Senegal": "🇸🇳", "Nigeria": "🇳🇬", "Egypt": "🇪🇬",
    "Cameroon": "🇨🇲", "South Africa": "🇿🇦", "Ghana": "🇬🇭", "Tunisia": "🇹🇳",
    "Ivory Coast": "🇨🇮", "Côte d'Ivoire": "🇨🇮", "Algeria": "🇩🇿",
    "Mali": "🇲🇱", "DR Congo": "🇨🇩", "Angola": "🇦🇴", "Zambia": "🇿🇲",
    "Cape Verde Islands": "🇨🇻", "Cape Verde": "🇨🇻", "Mozambique": "🇲🇿",
    "Japan": "🇯🇵", "South Korea": "🇰🇷", "Australia": "🇦🇺", "Iran": "🇮🇷",
    "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦", "Jordan": "🇯🇴", "UAE": "🇦🇪",
    "Uzbekistan": "🇺🇿", "Indonesia": "🇮🇩", "Iraq": "🇮🇶", "Oman": "🇴🇲",
    "China": "🇨🇳", "Bahrain": "🇧🇭", "Palestine": "🇵🇸", "India": "🇮🇳",
    "New Zealand": "🇳🇿",
}


def get_flag(team_name: str) -> str:
    if not team_name:
        return team_name
    if not team_name[0].isascii():
        return team_name
    clean = team_name.strip()
    if clean in _FLAGS:
        return f"{_FLAGS[clean]} {clean}"
    for key, flag in _FLAGS.items():
        if key.lower() in clean.lower():
            return f"{flag} {clean}"
    return clean


def flag_short(team_name: str) -> str:
    flagged = get_flag(team_name)
    parts   = flagged.split()
    if not parts:
        return team_name
    if not parts[0].isascii() and len(parts) > 1:
        return f"{parts[0]} {parts[1]}"
    return parts[0]


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_history() -> pd.DataFrame:
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


_HIGH_EV_THRESHOLD = 0.20


def ev_badge(row: pd.Series) -> str:
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


# ── Hero Header ───────────────────────────────────────────────────────────────
now_str = datetime.utcnow().strftime("%d %b %Y · %H:%M UTC")
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">
    <span class="hero-dot"></span>
    Live · WC 2026
  </div>
  <h1 class="hero-title">
    <span class="accent">MUNDIAL</span> PREDICTOR <span class="accent">2026</span>
  </h1>
  <div class="hero-stack">
    <span class="hero-tag">Poisson</span>
    <span class="hero-tag">Monte Carlo</span>
    <span class="hero-tag">Claude AI</span>
    <span class="hero-tag">Kelly Criterion</span>
    <span class="hero-tag">API-Football</span>
  </div>
  <div class="hero-update">Last updated: {now_str}</div>
</div>
""", unsafe_allow_html=True)

# ── KPI Bento Cards ───────────────────────────────────────────────────────────
total    = len(hist)
correct  = int(hist["correct_result"].sum())  if total else 0
exact    = int(hist["exact_match"].sum())     if total else 0
pts_earn = int(hist["points_earned"].sum())   if total else 0
pts_poss = int(hist["points_possible"].sum()) if total else 0

pct_correct = f"{correct/total*100:.0f}%" if total else "—"
pct_exact   = f"{exact/total*100:.0f}%"   if total else "—"
pts_label   = f"{pts_earn}/{pts_poss} pts" if pts_poss else "—"

st.markdown(f"""
<div class="kpi-grid">

  <div class="kpi-card c-cyan">
    <div class="kpi-glow"></div>
    <div class="kpi-label">Predictions</div>
    <div class="kpi-value">{total}</div>
    <div class="kpi-sub">Total matches scored</div>
  </div>

  <div class="kpi-card c-teal">
    <div class="kpi-glow"></div>
    <div class="kpi-label">Correct Results</div>
    <div class="kpi-value">{pct_correct}</div>
    <div class="kpi-sub">{correct}/{total} right outcome</div>
  </div>

  <div class="kpi-card c-gold">
    <div class="kpi-glow"></div>
    <div class="kpi-label">Exact Score Hits</div>
    <div class="kpi-value">{pct_exact}</div>
    <div class="kpi-sub">{exact}/{total} perfect scoreline</div>
  </div>

  <div class="kpi-card c-purple">
    <div class="kpi-glow"></div>
    <div class="kpi-label">Points Earned</div>
    <div class="kpi-value">{pts_earn}</div>
    <div class="kpi-sub">{pts_label}</div>
  </div>

</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="div-grad">', unsafe_allow_html=True)

# ── Today's Predictions ──────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
  <span class="icon">📋</span>
  <span class="title">Today's Predictions</span>
  <span class="line"></span>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Scores are FDR-adjusted (vice-captain.com) and AI-calibrated (Claude Opus). "
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

st.markdown('<hr class="div-grad">', unsafe_allow_html=True)

# ── Market vs AI Simulation ───────────────────────────────────────────────────
_SIM_COLS = {"sim_p_home", "sim_p_draw", "sim_p_away",
             "market_p_home", "market_p_draw", "market_p_away"}
has_sim = not picks.empty and _SIM_COLS.issubset(set(picks.columns))

if has_sim:
    st.markdown("""
    <div class="section-header">
      <span class="icon">🔥</span>
      <span class="title">Market vs AI Simulation</span>
      <span class="line"></span>
    </div>
    """, unsafe_allow_html=True)
    st.caption(
        "Bookmaker implied probability (after overround removal) vs "
        "Monte Carlo simulation (10,000 Poisson draws). "
        "🔥 Edge > 5% = Value Bet · ⭐ Edge ≥ 20% = High-Confidence"
    )

    sim_df = picks.copy()
    sim_df["Match"] = sim_df.apply(
        lambda r: f"{get_flag(r['home_team'])} vs {get_flag(r['away_team'])}", axis=1
    )

    fig_sim    = go.Figure()
    OUTCOMES   = [("Home", "p_home"), ("Draw", "p_draw"), ("Away", "p_away")]
    MKT_COLORS = [CLR_CYAN_D, "#4A5374", "#774B3B"]
    SIM_COLORS = [CLR_CYAN,   CLR_PURPLE, CLR_AMBER]

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
        yaxis=dict(gridcolor=CLR_GRID, title="Probability (%)", range=[0, 90]),
        xaxis=dict(gridcolor=CLR_GRID, tickangle=-15),
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
    st.markdown('<hr class="div-grad">', unsafe_allow_html=True)

# ── Charts (only when history exists) ────────────────────────────────────────
if hist.empty:
    st.info("📭 No historical data yet. Scores are logged automatically each morning after matches finish.")
else:
    # Row 1: Cumulative points | Accuracy donut
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("""
        <div class="section-header">
          <span class="icon">📈</span>
          <span class="title">Cumulative Points</span>
          <span class="line"></span>
        </div>
        """, unsafe_allow_html=True)
        cum = hist.copy()
        cum["Earned"]   = cum["points_earned"].cumsum()
        cum["Possible"] = cum["points_possible"].cumsum()
        cum["label"]    = cum["date"].dt.strftime("%-d %b")

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cum["label"], y=cum["Earned"],
            name="Points Earned", mode="lines+markers",
            line=dict(color=CLR_CYAN, width=2.5), marker=dict(size=5, color=CLR_CYAN),
            fill="tozeroy", fillcolor=f"rgba(0,200,255,0.07)",
        ))
        fig_cum.add_trace(go.Scatter(
            x=cum["label"], y=cum["Possible"],
            name="Max Possible", mode="lines",
            line=dict(color=CLR_GRID, width=1.5, dash="dot"),
        ))
        fig_cum.update_layout(
            **PLOTLY_LAYOUT, height=280,
            legend=dict(orientation="h", y=1.12, x=0),
            yaxis=dict(gridcolor=CLR_GRID, zeroline=False),
            xaxis=dict(gridcolor=CLR_GRID),
        )
        st.plotly_chart(fig_cum, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown("""
        <div class="section-header">
          <span class="icon">🎯</span>
          <span class="title">Accuracy</span>
          <span class="line"></span>
        </div>
        """, unsafe_allow_html=True)
        exact_n   = exact
        correct_n = correct - exact
        wrong_n   = total - correct

        fig_donut = go.Figure(go.Pie(
            labels=["Exact Score", "Correct Result", "Wrong"],
            values=[exact_n, correct_n, wrong_n],
            hole=0.62,
            marker_colors=[CLR_GOLD, CLR_TEAL, CLR_RED],
            textfont=dict(size=12),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{pct_correct}</b><br><span style='font-size:11px'>hit rate</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#EEF2FF"),
        )
        fig_donut.update_layout(
            **PLOTLY_LAYOUT, height=280,
            showlegend=True,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<hr class="div-grad">', unsafe_allow_html=True)

    # Row 2: Strategy vs Result
    st.markdown("""
    <div class="section-header">
      <span class="icon">🔄</span>
      <span class="title">Strategy vs Result — Last 10 Matches</span>
      <span class="line"></span>
    </div>
    """, unsafe_allow_html=True)

    recent = hist.tail(10).copy()
    recent["Match"] = recent.apply(
        lambda r: f"{flag_short(r['home_team'])} vs {flag_short(r['away_team'])}", axis=1
    )

    fig_bar = go.Figure()
    for label, col, color in [
        ("Pred. Home",   "predicted_home", CLR_CYAN),
        ("Actual Home",  "actual_home",    CLR_CYAN_D),
        ("Pred. Away",   "predicted_away", CLR_AMBER),
        ("Actual Away",  "actual_away",    CLR_AMBER_D),
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
        yaxis=dict(gridcolor=CLR_GRID, dtick=1, title="Goals"),
        xaxis=dict(gridcolor=CLR_GRID, tickangle=-25),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<hr class="div-grad">', unsafe_allow_html=True)

    # Row 3: Points by Stage
    if hist["stage_en"].nunique() > 1:
        st.markdown("""
        <div class="section-header">
          <span class="icon">🏆</span>
          <span class="title">Points by Tournament Stage</span>
          <span class="line"></span>
        </div>
        """, unsafe_allow_html=True)

        stage_df = (
            hist.groupby("stage_en")[["points_earned", "points_possible"]]
            .sum()
            .reset_index()
            .sort_values("points_possible", ascending=False)
        )

        fig_stage = go.Figure()
        fig_stage.add_trace(go.Bar(
            name="Max Possible", x=stage_df["stage_en"], y=stage_df["points_possible"],
            marker_color=CLR_GRID,
        ))
        fig_stage.add_trace(go.Bar(
            name="Earned", x=stage_df["stage_en"], y=stage_df["points_earned"],
            marker_color=CLR_CYAN,
        ))
        fig_stage.update_layout(
            **PLOTLY_LAYOUT, height=260,
            barmode="overlay",
            legend=dict(orientation="h", y=1.12, x=0),
            yaxis=dict(gridcolor=CLR_GRID),
            xaxis=dict(gridcolor=CLR_GRID),
        )
        st.plotly_chart(fig_stage, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<hr class="div-grad">', unsafe_allow_html=True)

    # Full history table
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
<div class="footer">
  <div class="footer-logo">
    <span class="accent">MUNDIAL</span> PREDICTOR 2026
  </div>
  <div class="footer-powered">
    AI by Claude (Anthropic) &nbsp;·&nbsp; Data: The Odds API · API-Football · vice-captain.com
  </div>
  <div class="footer-links">
    <a href="https://www.anthropic.com" target="_blank">Anthropic</a>
    <a href="#">Subscribe to WhatsApp picks</a>
    <a href="#">Contact</a>
  </div>
  <div class="footer-legal">
    For entertainment purposes only · Not financial or betting advice · Gamble responsibly
  </div>
</div>
""", unsafe_allow_html=True)
