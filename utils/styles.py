import streamlit as st

# ── Shared CSS theme ──────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Page header gradient banner ── */
.page-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25);
}
.page-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.page-header p  { margin: 0.4rem 0 0; opacity: 0.88; font-size: 0.95rem; }

/* ── White info card ── */
.info-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 16px rgba(100, 116, 139, 0.1);
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}

/* ── Gradient metric card ── */
.metric-card {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: white;
    text-align: center;
}
.metric-card .value { font-size: 1.9rem; font-weight: 700; line-height: 1; }
.metric-card .label {
    font-size: 0.72rem; opacity: 0.82;
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.3rem;
}

/* ── Score badge ── */
.score-badge {
    display: block;
    font-size: 2.8rem;
    font-weight: 800;
    padding: 1rem 2rem;
    border-radius: 16px;
    text-align: center;
    letter-spacing: -0.02em;
}
.score-excellent { background: linear-gradient(135deg, #10b981, #059669); color: white; }
.score-good      { background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: white; }
.score-average   { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
.score-poor      { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }

/* ── Section divider title ── */
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1e293b;
    padding-bottom: 0.45rem;
    border-bottom: 2px solid #6366f1;
    margin-bottom: 1rem;
}

/* ── Insight callout ── */
.insight-box {
    background: #f0f9ff;
    border-left: 4px solid #0ea5e9;
    padding: 0.85rem 1.1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: #0c4a6e;
    line-height: 1.55;
}

/* ── Feature card (home page) ── */
.feature-card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08);
    border-top: 4px solid #6366f1;
    height: 100%;
    transition: box-shadow 0.2s;
}
.feature-card:hover { box-shadow: 0 8px 32px rgba(99, 102, 241, 0.18); }
.feature-card .icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
.feature-card h3    { font-size: 0.95rem; font-weight: 600; color: #1e293b; margin: 0 0 0.4rem; }
.feature-card p     { font-size: 0.82rem; color: #64748b; margin: 0; line-height: 1.5; }

/* ── Stat card (home page) ── */
.stat-card {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 14px;
    padding: 1.4rem;
    color: white;
    text-align: center;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.28);
}
.stat-card .stat-number { font-size: 2rem; font-weight: 800; line-height: 1; }
.stat-card .stat-label  {
    font-size: 0.72rem; opacity: 0.82;
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.35rem;
}

/* ── Tech pill ── */
.tech-pill {
    display: inline-block;
    background: #ede9fe;
    color: #5b21b6;
    border-radius: 50px;
    padding: 0.22rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.2rem;
    border: 1px solid #c4b5fd;
}

/* ── Status badges ── */
.status-ready   { background: #d1fae5; color: #065f46; border-radius: 8px; padding: 0.5rem 1.1rem; font-size: 0.9rem; font-weight: 500; display: inline-block; }
.status-warning { background: #fef3c7; color: #92400e; border-radius: 8px; padding: 0.5rem 1.1rem; font-size: 0.9rem; font-weight: 500; display: inline-block; }

/* ── Force sidebar to always be visible ── */
section[data-testid="stSidebar"]   { display: flex !important; visibility: visible !important; }
[data-testid="collapsedControl"]   { display: block !important; visibility: visible !important; }
[data-testid="stSidebarNav"]       { display: block !important; }

/* ── Style page_link nav buttons in sidebar ── */
[data-testid="stSidebar"] [data-testid="stPageLink"] {
    border-radius: 8px;
    transition: background 0.15s;
}
[data-testid="stSidebar"] [data-testid="stPageLink"]:hover {
    background: #ede9fe;
}

/* ── Hide only deploy / share chrome ── */
#MainMenu                        { visibility: hidden; }
footer                           { visibility: hidden; }
[data-testid="stToolbar"]        { visibility: hidden; }
[data-testid="stDeployButton"]   { display: none; }
.stDeployButton                  { display: none; }
"""


def apply_theme():
    """Inject the shared CSS into the current page."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="page-header"><h1>{title}</h1>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def insight_box(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def sidebar_nav(active: str = ""):
    """Render a branded sidebar header + native page navigation links."""
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────────
        st.markdown(
            '<div style="text-align:center;padding:1rem 0 0.6rem;">'
            '<span style="font-size:2rem;">🎓</span><br>'
            '<span style="font-weight:700;font-size:1rem;color:#1e293b;">EduPredict AI</span><br>'
            '<span style="font-size:0.72rem;color:#94a3b8;">Student Score Prediction</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Native page links (clickable, highlighted on active page) ─────
        st.page_link("app.py",                              label="🏠  Home")
        st.page_link("pages/1_📊_Dashboard.py",          label="📊  Prediction Dashboard")
        st.page_link("pages/2_📈_Analytics.py",           label="📈  Data Analytics")
        st.page_link("pages/3_⚙️_Model_Details.py",       label="⚙️  Model Comparison")
        st.page_link("pages/4_Explainability.py",           label="🧠  SHAP Explainability")

        st.markdown("---")
        st.caption("v2.0 · Elevo Internship")


def metric_card(value, label):
    st.markdown(
        f'<div class="metric-card"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )
