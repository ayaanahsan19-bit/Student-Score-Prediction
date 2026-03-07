import streamlit as st
import os
import json

from utils.styles import apply_theme, sidebar_nav

st.set_page_config(
    page_title="EduPredict AI — Student Score Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
sidebar_nav("Home")

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header" style="text-align:center; padding: 3rem 2rem;">
    <div style="font-size:3.5rem; margin-bottom:0.4rem;">🎓</div>
    <h1 style="font-size:2.6rem; margin-bottom:0.5rem;">EduPredict AI</h1>
    <p style="font-size:1.05rem; opacity:0.92;">
        End-to-End Machine Learning Pipeline for Student Performance Prediction
    </p>
    <p style="font-size:0.82rem; opacity:0.7; margin-top:0.6rem;">
        Multi-model comparison &nbsp;·&nbsp; SHAP Explainability &nbsp;·&nbsp;
        Interactive Analytics &nbsp;·&nbsp; MLOps Ready
    </p>
</div>
""", unsafe_allow_html=True)

# ── Model status ──────────────────────────────────────────────────────────────
models_ready = os.path.exists("models/xgb_model.pkl")

if models_ready:
    st.markdown(
        '<div class="status-ready">✅ &nbsp; Models trained and ready — use the sidebar to navigate</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-warning">⚠️ &nbsp; Models not found — run <code>python train_pipeline.py</code> first</div>',
        unsafe_allow_html=True,
    )
    st.code("python train_pipeline.py", language="bash")

st.markdown("<br>", unsafe_allow_html=True)

# ── Live stat cards (only when models exist) ──────────────────────────────────
if models_ready:
    try:
        with open("models/metrics.json") as f:
            metrics = json.load(f)
        best_r2    = max(v["r2"]  for v in metrics.values())
        best_mse   = min(v["mse"] for v in metrics.values())
        best_model = max(metrics, key=lambda k: metrics[k]["r2"]).replace("_", " ")

        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (f"{best_r2:.1%}",   "Best R² Score"),
            (str(len(metrics)),  "Models Trained"),
            (f"{best_mse:.3f}",  "Lowest MSE"),
            ("3",                "Input Features"),
        ]
        for col, (val, lbl) in zip([c1, c2, c3, c4], cards):
            with col:
                st.markdown(
                    f'<div class="stat-card">'
                    f'<div class="stat-number">{val}</div>'
                    f'<div class="stat-label">{lbl}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)
    except Exception:
        pass

# ── Feature navigation cards ──────────────────────────────────────────────────
st.markdown('<div class="section-title">🚀 What You Can Explore</div>', unsafe_allow_html=True)

feature_cards = [
    ("📊", "Prediction Dashboard",
     "Tune student parameters with live sliders and get an instant XGBoost exam-score prediction with a dynamic gauge."),
    ("📈", "Data Analytics",
     "Interactive EDA: score distributions, correlation heatmaps, scatter plots, box plots, and summary stats."),
    ("⚙️", "Model Comparison",
     "Side-by-side R² / MSE comparison of Linear, Polynomial, Huber, and XGBoost with full residual diagnostics."),
    ("🧠", "SHAP Explainability",
     "Understand every prediction with global summary plots, waterfall charts, and feature dependence analysis."),
]

cols = st.columns(4)
for col, (icon, title, desc) in zip(cols, feature_cards):
    with col:
        st.markdown(
            f'<div class="feature-card">'
            f'<div class="icon">{icon}</div>'
            f'<h3>{title}</h3>'
            f'<p>{desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tech stack ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🛠️ Tech Stack</div>', unsafe_allow_html=True)
techs = ["Python 3.11", "Streamlit", "XGBoost", "Scikit-learn",
         "SHAP", "Plotly", "Pandas", "NumPy", "MLflow"]
st.markdown(
    " ".join(f'<span class="tech-pill">{t}</span>' for t in techs),
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── How to run section ────────────────────────────────────────────────────────
with st.expander("⚡ Quick Start Guide", expanded=False):
    st.markdown("""
    **Step 1 — Train the models** (run once):
    ```bash
    python train_pipeline.py
    ```
    **Step 2 — Launch the app**:
    ```bash
    streamlit run app.py
    ```
    **Step 3 — Explore!**
    Navigate the pages in the sidebar to predict scores, analyse data, compare models, and interpret predictions.
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:1.5rem; background:#f8fafc;
            border-radius:12px; color:#64748b; font-size:0.85rem; margin-top:1rem;">
    Built with ❤️ by <strong>Ayaan Ahsan</strong> &nbsp;|&nbsp;
    Elevo Internship Task &nbsp;|&nbsp; Powered by <strong>Streamlit</strong>
</div>
""", unsafe_allow_html=True)