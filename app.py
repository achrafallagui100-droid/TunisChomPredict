# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TunisChomPredict · app.py                                                 ║
# ║  Graduate Unemployment Risk Dashboard — Tunisia                             ║
# ║  Student: Achref Allegui | GR5 DS                                          ║
# ║  Run: streamlit run app.py                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TunisChomPredict",
    page_icon="🇹🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:        #07090f;
  --surface:   #0d1117;
  --surface2:  #131a24;
  --border:    #1e2d3d;
  --accent:    #3ecf8e;
  --accent2:   #6366f1;
  --warn:      #f59e0b;
  --danger:    #ef4444;
  --text:      #e2e8f0;
  --text-dim:  #64748b;
  --text-muted:#334155;
}

*, html, body { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Streamlit header ── */
[data-testid="stHeader"] {
  background: rgba(7,9,15,0.95) !important;
  border-bottom: 1px solid var(--border) !important;
}
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden !important; }
.block-container { padding-top: 1.8rem !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div { background: var(--border) !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.2rem 1.5rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: var(--accent2); }
[data-testid="stMetricLabel"] {
  font-size: 0.63rem !important;
  letter-spacing: 0.16em !important;
  text-transform: uppercase !important;
  color: var(--text-dim) !important;
}
[data-testid="stMetricValue"] {
  font-size: 2rem !important;
  font-weight: 700 !important;
  color: var(--text) !important;
}

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.4rem 1.6rem;
  margin: 0.6rem 0;
}
.card-low   { border-left: 4px solid var(--accent);  background: linear-gradient(135deg,rgba(62,207,142,.05),transparent); }
.card-mod   { border-left: 4px solid var(--warn);    background: linear-gradient(135deg,rgba(245,158,11,.05),transparent); }
.card-crit  { border-left: 4px solid var(--danger);  background: linear-gradient(135deg,rgba(239,68,68,.05),transparent); }

.badge-low  { background:#052e16; color:#4ade80; border:1px solid #16a34a; padding:4px 14px; border-radius:999px; font-size:.82rem; font-weight:600; }
.badge-mod  { background:#451a03; color:#fbbf24; border:1px solid #d97706; padding:4px 14px; border-radius:999px; font-size:.82rem; font-weight:600; }
.badge-crit { background:#1f0e0e; color:#f87171; border:1px solid #dc2626; padding:4px 14px; border-radius:999px; font-size:.82rem; font-weight:600; }

.val-display {
  font-family: 'JetBrains Mono', monospace;
  font-size: 3.2rem;
  font-weight: 700;
  line-height: 1;
  letter-spacing: -0.02em;
}
.label-sm {
  font-size: 0.62rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin: 1.4rem 0 0.4rem;
  display: block;
}
.hero-title {
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.hero-title .hl { color: var(--accent); }
.hero-sub { font-size: 0.80rem; color: var(--text-dim); margin-top: 0.35rem; letter-spacing: 0.05em; }
.footer-bar {
  text-align: center;
  padding: 1.6rem 0 0.6rem;
  color: var(--text-muted);
  font-size: 0.73rem;
  letter-spacing: 0.06em;
  border-top: 1px solid var(--border);
  margin-top: 2rem;
}
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ASSETS (cached)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    'year', 'quarter_num', 'gender_encoded', 'education_encoded',
    'is_national', 'is_post_revolution', 'is_covid_period',
    'is_recent', 'metric_encoded', 'gouvernorat_encoded', 'chomscore'
]

@st.cache_resource
def load_models():
    reg = pickle.load(open('models/xgboost_regressor.pkl', 'rb'))
    clf = pickle.load(open('models/xgboost_classifier.pkl', 'rb'))
    le  = pickle.load(open('models/label_encoder_gouvernorat.pkl', 'rb'))
    return reg, clf, le

@st.cache_data
def load_chomscore_table():
    cs = pd.read_csv('models/chomscore_table.csv', encoding='latin1')
    cs.columns = [c.strip().lower() for c in cs.columns]
    # build lookup: gouvernorat -> chomscore
    lookup = {}
    for _, row in cs.iterrows():
        lookup[str(row['gouvernorat']).strip()] = float(row['chomscore'])
    return cs, lookup

@st.cache_data
def load_history():
    df = pd.read_csv('tunisia_unemployment_4datasets_augmented.csv')
    return df

# ─────────────────────────────────────────────────────────────────────────────
# BOOT
# ─────────────────────────────────────────────────────────────────────────────
MODEL_OK = False
MODEL_ERR = ''
reg = clf = le = None

try:
    reg, clf, le = load_models()
    MODEL_OK = True
except Exception as e:
    MODEL_ERR = str(e)

cs_df, cs_lookup = load_chomscore_table()
hist_df = load_history()

# LE classes → display names (strip 'national')
if le is not None:
    GOUVERNORATS = [g for g in le.classes_.tolist() if g != 'national']
else:
    GOUVERNORATS = [
        "Ben Arous","Bizerte","Béja","Gabès","Gafsa","Jendouba",
        "Kairouan","Kasserine","Kef","Kébili","L'Ariana","Mahdia",
        "Manouba","Medenine","Monastir","Nabeul","Sfax","Sidi Bouzid",
        "Siliana","Sousse","Tataouine","Tozeur","Tunis","Zaghouan"
    ]

RISK_LABEL = {0: "Low", 1: "Moderate", 2: "Critical"}
RISK_BADGE = {
    0: "<span class='badge-low'>🟢 Low Risk</span>",
    1: "<span class='badge-mod'>🟡 Moderate Risk</span>",
    2: "<span class='badge-crit'>🔴 Critical Risk</span>",
}
RISK_COLOR = {0: "#4ade80", 1: "#fbbf24", 2: "#f87171"}
CARD_CLASS = {0: "card-low", 1: "card-mod", 2: "card-crit"}

PLOTLY_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#64748b', family='Inter', size=11),
    margin=dict(t=28, b=8, l=8, r=8),
)
CHART_CFG = {'displayModeBar': False}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: get chomscore for a gouvernorat
# ─────────────────────────────────────────────────────────────────────────────
def get_chomscore(gov: str) -> float:
    """Try an exact lookup then fuzzy on first word."""
    if gov in cs_lookup:
        return cs_lookup[gov]
    # Try stripping accents via simple mapping
    for k, v in cs_lookup.items():
        if k.lower().replace("'", "").replace(" ", "") == gov.lower().replace("'", "").replace(" ", ""):
            return v
    return 50.0  # fallback

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def predict(year, quarter_num, gender_encoded, education_encoded,
            gouvernorat_encoded, chomscore):
    # metric_encoded=1 → rate_pct  (= unemployment rate %)
    # metric_encoded=0 → count_thousands (raw count, NOT a percentage!)
    is_national        = 0
    is_post_revolution = 1 if year > 2010 else 0
    is_covid_period    = 1 if year in [2020, 2021] else 0
    is_recent          = 1 if year >= 2022 else 0
    metric_encoded     = 1   # rate_pct

    X = pd.DataFrame([{
        'year':               year,
        'quarter_num':        quarter_num,
        'gender_encoded':     gender_encoded,
        'education_encoded':  education_encoded,
        'is_national':        is_national,
        'is_post_revolution': is_post_revolution,
        'is_covid_period':    is_covid_period,
        'is_recent':          is_recent,
        'metric_encoded':     metric_encoded,
        'gouvernorat_encoded': gouvernorat_encoded,
        'chomscore':          chomscore,
    }], columns=FEATURES)

    raw_value  = float(reg.predict(X)[0])
    # Clamp to realistic unemployment rate range [0 – 60 %]
    pred_value = float(np.clip(raw_value, 0.0, 60.0))
    risk_class = int(clf.predict(X)[0])
    return pred_value, risk_class

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.8rem 0 0.5rem;'>
      <div style='font-size:1.35rem;font-weight:800;color:#e2e8f0;letter-spacing:-0.01em;'>
        🇹🇳 TunisChomPredict
      </div>
      <div style='font-size:0.67rem;color:#334155;letter-spacing:0.1em;text-transform:uppercase;margin-top:3px;'>
        Graduate Unemployment · Tunisia
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Model status ──
    if MODEL_OK:
        st.markdown("<div style='font-size:0.68rem;color:#4ade80;'>✅ Models loaded successfully</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='font-size:0.68rem;color:#f87171;'>❌ Model error: {MODEL_ERR[:80]}</div>",
                    unsafe_allow_html=True)
    st.divider()

    st.markdown("<span class='label-sm'>📍 Gouvernorat</span>", unsafe_allow_html=True)
    gouvernorat = st.selectbox("Gouvernorat", GOUVERNORATS, label_visibility="collapsed")

    st.markdown("<span class='label-sm'>👤 Gender</span>", unsafe_allow_html=True)
    gender_map = {"Total": 0, "Male": 1, "Female": 2}
    gender_sel = st.selectbox("Gender", list(gender_map.keys()), label_visibility="collapsed")
    gender_encoded = gender_map[gender_sel]

    st.markdown("<span class='label-sm'>🎓 Education Level</span>", unsafe_allow_html=True)
    edu_map = {"All Levels": 0, "Higher Education": 1}
    edu_sel = st.selectbox("Education", list(edu_map.keys()), label_visibility="collapsed")
    education_encoded = edu_map[edu_sel]

    st.markdown("<span class='label-sm'>📅 Year</span>", unsafe_allow_html=True)
    year = st.slider("Year", min_value=2007, max_value=2026, value=2024, label_visibility="collapsed")

    st.markdown("<span class='label-sm'>📆 Quarter</span>", unsafe_allow_html=True)
    quarter_map = {"Q1 (Jan–Mar)": 1, "Q2 (Apr–Jun)": 2, "Q3 (Jul–Sep)": 3, "Q4 (Oct–Dec)": 4}
    quarter_sel = st.selectbox("Quarter", list(quarter_map.keys()), label_visibility="collapsed")
    quarter_num = quarter_map[quarter_sel]

    st.divider()

    # Auto-derived flags info
    is_post  = 1 if year > 2010 else 0
    is_covid = 1 if year in [2020, 2021] else 0
    is_rec   = 1 if year >= 2022 else 0
    st.markdown(f"""
    <div style='font-size:0.70rem;color:#64748b;line-height:1.9;'>
      <b style='color:#94a3b8;'>Auto-derived flags</b><br/>
      Post-Revolution &nbsp;→ <code style='color:#3ecf8e;'>{is_post}</code><br/>
      COVID Period &nbsp;&nbsp;&nbsp;&nbsp;→ <code style='color:#3ecf8e;'>{is_covid}</code><br/>
      Recent (≥2022) &nbsp;&nbsp;→ <code style='color:#3ecf8e;'>{is_rec}</code><br/>
      Scope &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ <code style='color:#3ecf8e;'>Regional</code>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style='font-size:0.62rem;color:#1e293b;line-height:1.7;'>
      Source: INS Tunisia · Dataset augmented<br/>
      Models: XGBoost Regressor + Classifier
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────────────────────────────────────
gov_chomscore = get_chomscore(gouvernorat)

if le is not None:
    classes_list = le.classes_.tolist()
    gov_enc = classes_list.index(gouvernorat) if gouvernorat in classes_list else 0
else:
    gov_enc = 0

if MODEL_OK:
    pred_value, risk_class = predict(
        year, quarter_num, gender_encoded, education_encoded,
        gov_enc, gov_chomscore
    )
else:
    # Heuristic fallback
    pred_value = gov_chomscore / 100 * 35
    risk_class = 2 if gov_chomscore > 70 else (1 if gov_chomscore > 40 else 0)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1.1])
with col_h1:
    st.markdown(f"""
    <div class='hero-title'>Graduate Unemployment <span class='hl'>Risk</span> Dashboard</div>
    <div class='hero-sub'>
      🇹🇳 {gouvernorat.upper()} &nbsp;·&nbsp; {gender_sel.upper()} &nbsp;·&nbsp;
      {edu_sel.upper()} &nbsp;·&nbsp; {year} Q{quarter_num}
    </div>
    """, unsafe_allow_html=True)

with col_h2:
    rc = RISK_COLOR[risk_class]
    st.markdown(f"""
    <div style='text-align:right;'>
      <div style='font-size:0.62rem;letter-spacing:0.12em;color:#334155;text-transform:uppercase;'>Predicted Unemployment Rate</div>
      <div class='val-display' style='color:{rc};'>{pred_value:.1f}<span style='font-size:1.4rem;'>%</span></div>
      <div style='margin-top:6px;'>{RISK_BADGE[risk_class]}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS — 3 main metrics
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3 = st.columns(3)
with k1:
    # Tunisia national unemployment baseline ≈ 16 – 17 %  (INS 2024)
    BASELINE_RATE = 16.5
    delta_pp = pred_value - BASELINE_RATE
    st.metric(
        "📊 Predicted Unemployment Rate",
        f"{pred_value:.1f}%",
        delta=f"{delta_pp:+.1f} pp vs national baseline ({BASELINE_RATE}%)",
        delta_color="inverse",
    )
with k2:
    risk_icons = {0: "🟢", 1: "🟡", 2: "🔴"}
    st.metric(
        "🎯 Risk Category",
        f"{risk_icons[risk_class]} {RISK_LABEL[risk_class]}",
        delta=f"XGBoost classifier · class {risk_class}/2",
        delta_color="off",
    )
with k3:
    cs_nat_mean = float(cs_df[cs_df['gouvernorat'] != 'national']['chomscore'].mean())
    delta_cs = gov_chomscore - cs_nat_mean
    st.metric(
        "📈 ChomScore",
        f"{gov_chomscore:.1f} / 100",
        delta=f"{delta_cs:+.1f} vs national avg ({cs_nat_mean:.1f})",
        delta_color="inverse",
    )

st.markdown("<br/>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RISK INSIGHT CARD
# ─────────────────────────────────────────────────────────────────────────────
# Relocation targets: exclude current gouvernorat from the list
_reloc_targets = [g for g in ['Tunis','Sfax',"L'Ariana",'Ben Arous','Sousse'] if g != gouvernorat]
_reloc_str = ', '.join(_reloc_targets)

insights = {
    0: {
        "title": "🟢 Low Unemployment Risk",
        "body": f"""<b>{gouvernorat}</b> shows a <b>low unemployment risk</b> profile for
<b>{gender_sel}</b> graduates with <b>{edu_sel}</b> in <b>{year}</b>.<br/>
XGBoost predicted rate: <b>{pred_value:.1f}%</b>
(model trained on rate_pct metric, range 8.8–34.7%)
· ChomScore: <b>{gov_chomscore:.1f}/100</b>
<ul>
  <li>Target high-growth sectors: digital services, fintech, renewable energy, agritech</li>
  <li>Build trilingual portfolio (Arabic / French / English) on GitHub + LinkedIn</li>
  <li>Apply to international VIE programs (France, Germany, Canada)</li>
  <li>Leverage Upwork / Toptal – Tunisia's cost advantage in freelance markets</li>
</ul>
"""
    },
    1: {
        "title": "🟡 Moderate Unemployment Risk",
        "body": f"""<b>{gouvernorat}</b> shows a <b>moderate unemployment risk</b> for
<b>{gender_sel}</b> graduates with <b>{edu_sel}</b> in <b>{year}</b>.<br/>
XGBoost predicted rate: <b>{pred_value:.1f}%</b>
· ChomScore: <b>{gov_chomscore:.1f}/100</b>
<ul>
  <li>Short-cycle IT certifications: AWS Cloud Practitioner, Meta Front-End Developer</li>
  <li>ANETI job placement centers — free coaching + employer matching</li>
  <li>Explore opportunities in: {_reloc_str}</li>
  <li>Vocational tourism &amp; hospitality programs (6–12 months) — ONTT accredited</li>
</ul>
"""
    },
    2: {
        "title": "🔴 Critical Unemployment Risk",
        "body": f"""<b>{gouvernorat}</b> shows a <b>critical unemployment risk</b> for
<b>{gender_sel}</b> graduates with <b>{edu_sel}</b> in <b>{year}</b>.<br/>
XGBoost predicted rate: <b>{pred_value:.1f}%</b>
· ChomScore: <b>{gov_chomscore:.1f}/100</b>
<ul>
  <li>Entrepreneurship grants → BFPME micro-loans, FONAPRAM startup fund</li>
  <li>Relocate job search to: {_reloc_str}</li>
  <li>CISCO NetAcad / AWS / Meta certifications for rapid IT entry</li>
  <li>Public sector exam preparation (concours nationaux) — additional pathway</li>
</ul>
"""
    },
}

ins = insights[risk_class]
st.markdown(f"""
<div class='card {CARD_CLASS[risk_class]}'>
  <div style='font-size:1.05rem;font-weight:700;margin-bottom:.5rem;'>{ins['title']}</div>
  <div style='font-size:0.84rem;line-height:1.75;color:#cbd5e1;'>
    {ins['body']}
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS ROW 1: Bar chart all gouvernorats ChomScore + Line chart history
# ─────────────────────────────────────────────────────────────────────────────
col_a, col_b = st.columns([1.2, 1])

# ── Chart A: ChomScore for all gouvernorats ──
with col_a:
    st.markdown("<span class='label-sm'>📊 ChomScore — All Gouvernorats</span>", unsafe_allow_html=True)
    bar_df = cs_df[cs_df['gouvernorat'] != 'national'].copy()
    bar_df = bar_df.sort_values('chomscore', ascending=True)

    colors_bar = [
        '#3ecf8e' if g == gouvernorat
        else ('#ef4444' if s > 94 else '#1e3a5f')
        for g, s in zip(bar_df['gouvernorat'], bar_df['chomscore'])
    ]

    fig_bar = go.Figure(go.Bar(
        x=bar_df['chomscore'].round(1),
        y=bar_df['gouvernorat'],
        orientation='h',
        marker=dict(color=colors_bar, line=dict(color='rgba(0,0,0,0)', width=0)),
        text=bar_df['chomscore'].round(1),
        textposition='outside',
        textfont=dict(color='#64748b', size=9),
    ))
    nat_mean = float(cs_df[cs_df['gouvernorat'] != 'national']['chomscore'].mean())
    fig_bar.add_vline(
        x=nat_mean,
        line=dict(color='#6366f1', width=1.5, dash='dot'),
        annotation_text=f'Mean {nat_mean:.1f}',
        annotation_font=dict(color='#6366f1', size=9),
    )
    fig_bar.update_layout(
        **{**PLOTLY_BASE, 'margin': dict(t=20, b=8, l=8, r=55)},
        height=max(400, len(bar_df) * 18),
        xaxis=dict(gridcolor='#1e2d3d', range=[0, 115], zeroline=False),
        yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=10)),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, width='stretch', config=CHART_CFG)

# ── Chart B: Historical unemployment trend for selected gouvernorat ──
with col_b:
    st.markdown(
        f"<span class='label-sm'>📈 Historical Trend · {gouvernorat} (Gender = Total)</span>",
        unsafe_allow_html=True,
    )
    # Filter: selected gouvernorat, gender = total (0), metric = rate_pct
    trend = hist_df[
        (hist_df['gouvernorat'] == gouvernorat) &
        (hist_df['gender_encoded'] == 0) &
        (hist_df['metric'] == 'rate_pct')
    ].copy()

    if trend.empty:
        # Try without metric filter
        trend = hist_df[
            (hist_df['gouvernorat'] == gouvernorat) &
            (hist_df['gender_encoded'] == 0)
        ].copy()

    if not trend.empty:
        trend = trend.sort_values('year')
        trend_agg = trend.groupby('year')['value'].mean().reset_index()

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=trend_agg['year'],
            y=trend_agg['value'],
            mode='lines+markers',
            name=gouvernorat,
            line=dict(color='#3ecf8e', width=2.5),
            marker=dict(size=6, color='#3ecf8e',
                        line=dict(color='#07090f', width=1.5)),
            fill='tozeroy',
            fillcolor='rgba(62,207,142,0.07)',
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
        ))
        # Highlight selected year
        yr_val = trend_agg[trend_agg['year'] == year]['value']
        if not yr_val.empty:
            fig_line.add_trace(go.Scatter(
                x=[year], y=[float(yr_val.iloc[0])],
                mode='markers',
                name=f'Selected ({year})',
                marker=dict(size=12, color='#f59e0b', symbol='diamond',
                            line=dict(color='#07090f', width=2)),
            ))
        fig_line.update_layout(
            **PLOTLY_BASE,
            height=400,
            xaxis=dict(gridcolor='#1e2d3d', zeroline=False,
                       dtick=2, tickangle=-45),
            yaxis=dict(gridcolor='#1e2d3d', zeroline=False,
                       title=dict(text='Rate (%)', font=dict(size=10))),
            legend=dict(orientation='h', y=-0.18, font=dict(size=10)),
            showlegend=True,
        )
    else:
        fig_line = go.Figure()
        fig_line.add_annotation(
            text=f"No historical data for {gouvernorat}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color='#64748b', size=13),
        )
        fig_line.update_layout(**PLOTLY_BASE, height=400)

    st.plotly_chart(fig_line, width='stretch', config=CHART_CFG)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS ROW 2: Risk class probability gauge + Gender comparison bar
# ─────────────────────────────────────────────────────────────────────────────
col_c, col_d = st.columns(2)

# ── Gauge: ChomScore vs thresholds ──
with col_c:
    st.markdown("<span class='label-sm'>🎯 ChomScore Gauge</span>", unsafe_allow_html=True)
    gauge_color = RISK_COLOR[risk_class]
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gov_chomscore,
        number={'suffix': '/100', 'font': {'color': '#e2e8f0', 'family': 'JetBrains Mono', 'size': 34}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#1e2d3d', 'tickwidth': 1},
            'bar': {'color': gauge_color, 'thickness': 0.25},
            'bgcolor': '#0d1117',
            'bordercolor': '#1e2d3d',
            'steps': [
                {'range': [0, 40],   'color': 'rgba(74,222,128,0.07)'},
                {'range': [40, 70],  'color': 'rgba(251,191,36,0.06)'},
                {'range': [70, 100], 'color': 'rgba(248,113,113,0.09)'},
            ],
            'threshold': {
                'line': {'color': '#6366f1', 'width': 2},
                'thickness': 0.8,
                'value': nat_mean,
            },
        },
    ))
    fig_gauge.update_layout(**PLOTLY_BASE, height=280)
    st.plotly_chart(fig_gauge, width='stretch', config=CHART_CFG)

# ── Bar: predicted value across genders for current gouvernorat/year ──
with col_d:
    st.markdown(
        f"<span class='label-sm'>👥 Predicted Rate by Gender · {gouvernorat} {year}</span>",
        unsafe_allow_html=True,
    )
    if MODEL_OK:
        genders_display = ["Total", "Male", "Female"]
        gen_preds = []
        for g_enc, g_label in enumerate(genders_display):
            pv, _ = predict(year, quarter_num, g_enc, education_encoded, gov_enc, gov_chomscore)
            gen_preds.append({'Gender': g_label, 'Rate (%)': round(pv, 2)})
        gen_df = pd.DataFrame(gen_preds)
        colors_gen = ['#3ecf8e', '#6366f1', '#f59e0b']
        fig_gen = go.Figure(go.Bar(
            x=gen_df['Gender'],
            y=gen_df['Rate (%)'],
            marker=dict(
                color=colors_gen,
                line=dict(color='rgba(0,0,0,0)', width=0),
            ),
            text=gen_df['Rate (%)'].apply(lambda v: f'{v:.2f}%'),
            textposition='outside',
            textfont=dict(color='#94a3b8', size=11),
        ))
        fig_gen.update_layout(
            **PLOTLY_BASE,
            height=280,
            yaxis=dict(gridcolor='#1e2d3d', zeroline=False),
            xaxis=dict(gridcolor='rgba(0,0,0,0)'),
            showlegend=False,
        )
        st.plotly_chart(fig_gen, width='stretch', config=CHART_CFG)
    else:
        st.info("Models not loaded — gender comparison unavailable.")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 🗺️  CHOROPLETH BUBBLE MAP — ChomScore by Gouvernorat
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("<span class='label-sm'>🗺️ Tunisia ChomScore Map — Regional Risk Overview</span>",
            unsafe_allow_html=True)

# Approximate centroids for Tunisia's 24 gouvernorats
GOV_COORDS = {
    "Tunis":       (36.819,  10.166),
    "Sfax":        (34.740,  10.760),
    "Sousse":      (35.825,  10.641),
    "Monastir":    (35.783,  10.826),
    "Nabeul":      (36.456,  10.738),
    "L'Ariana":    (36.867,  10.165),
    "Ben Arous":   (36.750,  10.226),
    "Manouba":     (36.809,  10.001),
    "Bizerte":     (37.274,   9.874),
    "Béja":        (36.726,   9.182),
    "Jendouba":    (36.501,   8.781),
    "Kef":         (36.183,   8.715),
    "Siliana":     (36.085,   9.374),
    "Zaghouan":    (36.403,  10.143),
    "Kairouan":    (35.678,  10.097),
    "Kasserine":   (35.172,   8.831),
    "Sidi Bouzid": (35.038,   9.485),
    "Mahdia":      (35.503,  11.062),
    "Gabès":       (33.888,  10.098),
    "Medenine":    (33.340,  10.506),
    "Gafsa":       (34.417,   8.783),
    "Tozeur":      (33.920,   8.134),
    "Kébili":      (33.703,   8.969),
    "Tataouine":   (32.930,  10.452),
}

# Build map dataframe — match cs_df gouvernorat names to coord dict with fuzzy fallback
map_rows = []
for _, row in cs_df[cs_df['gouvernorat'] != 'national'].iterrows():
    g_raw = str(row['gouvernorat']).strip()
    coords = None
    # Exact match first
    if g_raw in GOV_COORDS:
        coords = GOV_COORDS[g_raw]
    else:
        # Fuzzy: strip accents/apostrophes
        g_key = g_raw.lower().replace("'", "").replace(" ", "")
        for k, v in GOV_COORDS.items():
            if k.lower().replace("'", "").replace(" ", "") == g_key:
                coords = v
                break
    if coords:
        score = float(row['chomscore'])
        risk  = str(row.get('risk', 'Low Risk'))
        # Compute predicted rate for this gouvernorat at current sidebar settings
        if MODEL_OK and g_raw in [g for g in le.classes_.tolist() if g != 'national']:
            cl = le.classes_.tolist()
            ge = cl.index(g_raw) if g_raw in cl else 0
            _cs = get_chomscore(g_raw)
            _pv, _rc = predict(year, quarter_num, gender_encoded, education_encoded, ge, _cs)
        else:
            _pv, _rc = score / 100 * 25, (2 if score > 70 else (1 if score > 40 else 0))
        map_rows.append({
            'Gouvernorat': g_raw,
            'lat': coords[0],
            'lon': coords[1],
            'ChomScore': score,
            'PredRate': round(_pv, 1),
            'Risk': risk,
            'RiskClass': _rc,
            'IsSelected': g_raw == gouvernorat,
        })

map_df = pd.DataFrame(map_rows)

if not map_df.empty:
    # Color by risk class
    _color_map = {0: '#4ade80', 1: '#fbbf24', 2: '#f87171'}
    map_df['Color']    = map_df['RiskClass'].map(_color_map).fillna('#64748b')
    map_df['Size']     = map_df['ChomScore'].apply(lambda s: max(s * 0.6, 8))
    map_df['Opacity']  = map_df['IsSelected'].apply(lambda x: 1.0 if x else 0.72)

    fig_map = go.Figure()

    # Non-selected gouvernorats
    for rc_val, rc_label, rc_col in [(0,'Low','#4ade80'),(1,'Moderate','#fbbf24'),(2,'Critical','#f87171')]:
        sub = map_df[(map_df['RiskClass'] == rc_val) & (~map_df['IsSelected'])]
        if not sub.empty:
            fig_map.add_trace(go.Scattergeo(
                lat=sub['lat'], lon=sub['lon'],
                mode='markers+text',
                name=f'{rc_label} Risk',
                marker=dict(
                    size=sub['Size'],
                    color=rc_col,
                    opacity=0.70,
                    line=dict(color='#07090f', width=1),
                ),
                text=sub['Gouvernorat'],
                textposition='top center',
                textfont=dict(size=8, color='#94a3b8'),
                customdata=sub[['ChomScore','PredRate','Risk']].values,
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'ChomScore: %{customdata[0]:.1f}/100<br>'
                    'Predicted Rate: %{customdata[1]:.1f}%<br>'
                    'Risk: %{customdata[2]}<extra></extra>'
                ),
            ))

    # Selected gouvernorat — highlighted on top
    sel = map_df[map_df['IsSelected']]
    if not sel.empty:
        fig_map.add_trace(go.Scattergeo(
            lat=sel['lat'], lon=sel['lon'],
            mode='markers+text',
            name=f'▶ {gouvernorat}',
            marker=dict(
                size=sel['Size'] * 1.6,
                color='#ffffff',
                opacity=1.0,
                symbol='star',
                line=dict(color=RISK_COLOR[risk_class], width=3),
            ),
            text=sel['Gouvernorat'],
            textposition='top center',
            textfont=dict(size=10, color='#e2e8f0', family='Inter'),
            customdata=sel[['ChomScore','PredRate','Risk']].values,
            hovertemplate=(
                '<b>★ %{text}</b> (selected)<br>'
                'ChomScore: %{customdata[0]:.1f}/100<br>'
                'Predicted Rate: %{customdata[1]:.1f}%<br>'
                'Risk: %{customdata[2]}<extra></extra>'
            ),
        ))

    fig_map.update_geos(
        visible=True,
        resolution=50,
        scope='africa',
        showcoastlines=True, coastlinecolor='#1e2d3d',
        showland=True,       landcolor='#0d1117',
        showocean=True,      oceancolor='#07090f',
        showlakes=False,
        showcountries=True,  countrycolor='#1e2d3d',
        showsubunits=False,
        center=dict(lat=34.0, lon=9.5),
        projection_scale=5.5,
        lataxis_range=[30.0, 38.5],
        lonaxis_range=[7.0,  12.5],
        bgcolor='rgba(0,0,0,0)',
        framecolor='#1e2d3d',
    )
    fig_map.update_layout(
        **{**PLOTLY_BASE, 'margin': dict(t=10, b=5, l=5, r=5)},
        height=460,
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        legend=dict(
            orientation='v', x=1.01, y=0.98,
            font=dict(size=10), bgcolor='rgba(13,17,23,0.85)',
            bordercolor='#1e2d3d', borderwidth=1,
        ),
        showlegend=True,
    )
    st.plotly_chart(fig_map, width='stretch', config=CHART_CFG)

    st.caption(
        "Bubble size ∝ ChomScore · Color = XGBoost risk class · ★ = selected gouvernorat  "
        "· 🟢 Low  🟡 Moderate  🔴 Critical"
    )

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 📊 FEATURE IMPORTANCE — Regressor & Classifier
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("<span class='label-sm'>📊 Feature Importance — XGBoost Models</span>",
            unsafe_allow_html=True)

FEAT_LABELS = {
    'year':               'Year',
    'quarter_num':        'Quarter',
    'gender_encoded':     'Gender',
    'education_encoded':  'Education Level',
    'is_national':        'Is National',
    'is_post_revolution': 'Post-Revolution Flag',
    'is_covid_period':    'COVID Period Flag',
    'is_recent':          'Recent Period (≥2022)',
    'metric_encoded':     'Metric Type',
    'gouvernorat_encoded':'Gouvernorat (Region)',
    'chomscore':          'ChomScore Index',
}

if MODEL_OK:
    col_fi1, col_fi2 = st.columns(2)

    def _importance_fig(model_obj, title, bar_color):
        imps = model_obj.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': [FEAT_LABELS.get(f, f) for f in FEATURES],
            'Importance': imps,
        }).sort_values('Importance', ascending=True)

        # Highlight gouvernorat and chomscore
        colors_fi = [
            '#f59e0b' if 'Gouvernorat' in f or 'ChomScore' in f
            else bar_color
            for f in fi_df['Feature']
        ]

        fig_fi = go.Figure(go.Bar(
            x=fi_df['Importance'],
            y=fi_df['Feature'],
            orientation='h',
            marker=dict(color=colors_fi, line=dict(color='rgba(0,0,0,0)')),
            text=fi_df['Importance'].apply(lambda v: f'{v:.3f}'),
            textposition='outside',
            textfont=dict(size=9, color='#64748b'),
        ))
        fig_fi.update_layout(
            **{**PLOTLY_BASE, 'margin': dict(t=28, b=8, l=8, r=50)},
            title=dict(text=title, font=dict(size=12, color='#94a3b8'), x=0),
            height=340,
            xaxis=dict(gridcolor='#1e2d3d', zeroline=False, range=[0, max(imps)*1.25]),
            yaxis=dict(tickfont=dict(size=10)),
            showlegend=False,
        )
        return fig_fi

    with col_fi1:
        st.plotly_chart(
            _importance_fig(reg, 'XGBoost Regressor — Rate Prediction', '#3ecf8e'),
            width='stretch', config=CHART_CFG
        )
    with col_fi2:
        st.plotly_chart(
            _importance_fig(clf, 'XGBoost Classifier — Risk Category', '#6366f1'),
            width='stretch', config=CHART_CFG
        )

    # Research answer
    reg_imps = dict(zip(FEATURES, reg.feature_importances_))
    clf_imps = dict(zip(FEATURES, clf.feature_importances_))
    gov_reg = reg_imps['gouvernorat_encoded']
    gen_reg = reg_imps['gender_encoded']
    gov_clf = clf_imps['gouvernorat_encoded']
    gen_clf = clf_imps['gender_encoded']

    if gov_reg > gen_reg:
        finding = f"🏛️ <b>Gouvernorat</b> outweighs Gender in the regressor ({gov_reg:.3f} vs {gen_reg:.3f}) — geography is the stronger predictor."
    else:
        finding = f"👤 <b>Gender</b> outweighs Gouvernorat in the regressor ({gen_reg:.3f} vs {gov_reg:.3f}) — demographic factor is stronger."

    st.markdown(f"""
    <div class='card' style='padding:1rem 1.4rem;margin-top:.5rem;'>
      <div style='font-size:0.78rem;color:#94a3b8;line-height:1.8;'>
        <b style='color:#e2e8f0;'>Research Finding</b> (Proposal §4 — Feature Importance)<br/>
        Regressor → Gouvernorat: <code style='color:#f59e0b'>{gov_reg:.3f}</code> &nbsp;|&nbsp;
        Gender: <code style='color:#6366f1'>{gen_reg:.3f}</code><br/>
        Classifier → Gouvernorat: <code style='color:#f59e0b'>{gov_clf:.3f}</code> &nbsp;|&nbsp;
        Gender: <code style='color:#6366f1'>{gen_clf:.3f}</code><br/>
        {finding}
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("Models not loaded — feature importance unavailable.")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 📅  2026 QUARTERLY FORECAST — selected gouvernorat
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<span class='label-sm'>📅 2026 Quarterly Forecast · {gouvernorat} · {gender_sel} · {edu_sel}</span>",
    unsafe_allow_html=True,
)

if MODEL_OK:
    forecast_rows = []
    for q in [1, 2, 3, 4]:
        pv, rc = predict(2026, q, gender_encoded, education_encoded, gov_enc, gov_chomscore)
        forecast_rows.append({
            'Quarter': f'Q{q} 2026',
            'Predicted Rate (%)': round(pv, 2),
            'Risk Class': rc,
            'Risk Label': RISK_LABEL[rc],
            'Risk Color': RISK_COLOR[rc],
        })
    fc_df = pd.DataFrame(forecast_rows)

    col_fc1, col_fc2 = st.columns([1.6, 1])

    with col_fc1:
        # Line chart with colored markers per risk
        fig_fc = go.Figure()
        # Shaded area under line
        fig_fc.add_trace(go.Scatter(
            x=fc_df['Quarter'],
            y=fc_df['Predicted Rate (%)'],
            mode='lines',
            line=dict(color='#6366f1', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(99,102,241,0.07)',
            showlegend=False,
        ))
        # Colored markers per risk class
        for rc_val, rc_col, rc_lbl in [(0,'#4ade80','Low'),(1,'#fbbf24','Moderate'),(2,'#f87171','Critical')]:
            sub = fc_df[fc_df['Risk Class'] == rc_val]
            if not sub.empty:
                fig_fc.add_trace(go.Scatter(
                    x=sub['Quarter'],
                    y=sub['Predicted Rate (%)'],
                    mode='markers+text',
                    name=f'{rc_lbl} Risk',
                    marker=dict(size=14, color=rc_col,
                                line=dict(color='#07090f', width=2)),
                    text=sub['Predicted Rate (%)'].apply(lambda v: f'{v:.1f}%'),
                    textposition='top center',
                    textfont=dict(size=11, color=rc_col),
                ))
        # National baseline
        fig_fc.add_hline(
            y=16.5,
            line=dict(color='#64748b', width=1, dash='dot'),
            annotation_text='National baseline 16.5%',
            annotation_font=dict(color='#64748b', size=9),
        )
        fig_fc.update_layout(
            **PLOTLY_BASE,
            height=300,
            xaxis=dict(gridcolor='rgba(0,0,0,0)'),
            yaxis=dict(gridcolor='#1e2d3d', zeroline=False,
                       title=dict(text='Rate (%)', font=dict(size=10)),
                       range=[0, max(fc_df['Predicted Rate (%)'].max()*1.3, 30)]),
            legend=dict(orientation='h', y=-0.22, font=dict(size=10)),
        )
        st.plotly_chart(fig_fc, width='stretch', config=CHART_CFG)

    with col_fc2:
        st.markdown("<span class='label-sm'>Forecast Summary Table</span>", unsafe_allow_html=True)
        for _, row in fc_df.iterrows():
            badge_cls = {0:'badge-low', 1:'badge-mod', 2:'badge-crit'}[row['Risk Class']]
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:0.55rem 0.9rem;border-bottom:1px solid #1e2d3d;'>
              <span style='font-family:JetBrains Mono,monospace;font-size:0.85rem;
                           color:#94a3b8;'>{row['Quarter']}</span>
              <span style='font-family:JetBrains Mono,monospace;font-size:1rem;
                           color:{row["Risk Color"]};font-weight:700;'>
                {row['Predicted Rate (%)']:.1f}%
              </span>
              <span class='{badge_cls}' style='font-size:0.70rem;padding:2px 10px;'>
                {row['Risk Label']}
              </span>
            </div>
            """, unsafe_allow_html=True)

        # Average
        avg_rate = fc_df['Predicted Rate (%)'].mean()
        dominant_rc = int(fc_df['Risk Class'].mode()[0])
        dom_col = RISK_COLOR[dominant_rc]
        st.markdown(f"""
        <div style='padding:0.6rem 0.9rem;margin-top:0.4rem;
                    background:rgba(99,102,241,0.07);border-radius:8px;
                    border:1px solid rgba(99,102,241,0.2);'>
          <div style='font-size:0.65rem;color:#64748b;letter-spacing:.12em;
                      text-transform:uppercase;'>2026 Annual Average</div>
          <div style='font-size:1.5rem;font-weight:700;color:{dom_col};
                      font-family:JetBrains Mono,monospace;margin-top:2px;'>
            {avg_rate:.1f}%
          </div>
          <div style='font-size:0.72rem;color:#64748b;margin-top:2px;'>
            Dominant risk: {RISK_LABEL[dominant_rc]}
          </div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.warning("Models not loaded — 2026 forecast unavailable.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CHOMSCORE TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<span class='label-sm'>📋 ChomScore Table — All Gouvernorats</span>", unsafe_allow_html=True)
display_cs = cs_df[cs_df['gouvernorat'] != 'national'].copy()
display_cs = display_cs.sort_values('chomscore', ascending=False)
display_cs['chomscore'] = display_cs['chomscore'].round(1)
display_cs.columns = [c.capitalize() for c in display_cs.columns]
st.dataframe(
    display_cs,
    width='stretch',
    hide_index=True,
    column_config={
        'Gouvernorat': st.column_config.TextColumn("Gouvernorat", width="medium"),
        'Chomscore': st.column_config.ProgressColumn(
            "ChomScore", min_value=0, max_value=100, format="%.1f"
        ),
        'Risk': st.column_config.TextColumn("Risk Level", width="medium"),
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer-bar'>
  Student: <b>Achref Allegui</b> &nbsp;|&nbsp; GR5 DS
  &nbsp;&nbsp;·&nbsp;&nbsp;
  TunisChomPredict · Graduate Unemployment Risk · Tunisia 2026
</div>
""", unsafe_allow_html=True)

