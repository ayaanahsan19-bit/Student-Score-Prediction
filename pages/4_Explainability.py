import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.preprocessing import load_and_clean_data
from utils.styles import apply_theme, page_header, section_title, insight_box

st.set_page_config(page_title="SHAP Explainability", page_icon="🧠", layout="wide")
apply_theme()
page_header("🧠 Model Explainability (SHAP)",
            "Understand why the model makes each prediction using SHAP values")

# ── What is SHAP? ─────────────────────────────────────────────────────────────
with st.expander("ℹ️  What is SHAP?", expanded=False):
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** is a game-theoretic approach that explains
    the output of any machine learning model.

    | Plot | What it shows |
    |---|---|
    | **Mean \|SHAP\|** | Which features matter most globally |
    | **Summary (beeswarm)** | Direction and magnitude of each feature across all samples |
    | **Waterfall** | Why a *single* student got their predicted score |
    | **Dependence** | How one feature's value affects its SHAP contribution |
    """)

# ── Load model & data ─────────────────────────────────────────────────────────
try:
    df       = load_and_clean_data()
    features = joblib.load("models/features.pkl")
    model    = joblib.load("models/xgb_model.pkl")
    X_sample = df[features].sample(200, random_state=42).reset_index(drop=True)
except Exception as e:
    st.error(f"Models not found: {e}. Run `python train_pipeline.py` first.")
    st.stop()

# ── SHAP computation ──────────────────────────────────────────────────────────
with st.spinner("⚙️ Calculating SHAP values for 200 students …"):
    explainer   = shap.Explainer(model)
    shap_values = explainer(X_sample)

st.success(f"✅ SHAP values ready — analysing {len(X_sample)} sample students")
st.markdown("---")

# ── Row 1: Mean |SHAP| bar (Plotly) + Summary beeswarm (matplotlib/SHAP) ─────
col1, col2 = st.columns(2)

with col1:
    section_title("🏆 Global Feature Importance (Mean |SHAP|)")
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    feat_labels = [f.replace("_", " ") for f in features]
    sort_idx  = np.argsort(mean_shap)
    s_feats   = [feat_labels[i] for i in sort_idx]
    s_vals    = [mean_shap[i]   for i in sort_idx]

    fig_imp = go.Figure(go.Bar(
        x=s_vals, y=s_feats,
        orientation="h",
        marker=dict(color=s_vals, colorscale="Viridis", showscale=False),
        text=[f"{v:.4f}" for v in s_vals],
        textposition="outside",
    ))
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
        font={"family": "Inter"}, height=300,
        xaxis=dict(title="Mean |SHAP Value|", gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=55, t=20, b=10),
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    top_feat = s_feats[-1]
    insight_box(
        f"**{top_feat}** is the single most influential feature, contributing an average of "
        f"**{s_vals[-1]:.4f}** score points in absolute SHAP impact."
    )

with col2:
    section_title("🐝 SHAP Summary (Beeswarm)")
    fig_beeswarm = plt.figure(figsize=(7, 3.8))
    shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig_beeswarm, use_container_width=True)
    plt.close(fig_beeswarm)

st.markdown("---")

# ── Row 2: Waterfall for selected student ─────────────────────────────────────
section_title("🔍 Individual Student Explanation")

col3, col4 = st.columns([1, 2.2])

with col3:
    idx = int(st.number_input(
        "Select student index (0 – 199)",
        min_value=0, max_value=len(X_sample) - 1, value=0, step=1,
    ))
    student = X_sample.iloc[idx]
    predicted = float(model.predict(student.values.reshape(1, -1))[0])
    predicted = max(0.0, min(100.0, predicted))

    st.markdown("**Student Profile**")
    for feat in features:
        st.markdown(f"- **{feat.replace('_', ' ')}:** {student[feat]}")

    st.markdown(
        f'<div class="info-card" style="margin-top:1rem;text-align:center;">'
        f'<div style="font-size:1.7rem;font-weight:800;color:#6366f1;">{predicted:.1f}%</div>'
        f'<div style="font-size:.78rem;color:#64748b;text-transform:uppercase;'
        f'letter-spacing:.06em;">Predicted Score</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with col4:
    fig_wf = plt.figure(figsize=(9, 3.6))
    shap.waterfall_plot(shap_values[idx], show=False, max_display=10)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig_wf, use_container_width=True)
    plt.close(fig_wf)

insight_box(
    "The waterfall chart shows how each feature **pushes** the prediction up (red) "
    "or down (blue) from the model's base value."
)

st.markdown("---")

# ── Feature dependence plot (matplotlib/SHAP) ─────────────────────────────────
section_title("📈 Feature Dependence Plot")

selected_feat = st.selectbox(
    "Choose a feature to inspect:",
    options=features,
    format_func=lambda x: x.replace("_", " "),
)

fig_dep, ax_dep = plt.subplots(figsize=(10, 4))
shap.dependence_plot(
    selected_feat, shap_values.values, X_sample,
    show=False, ax=ax_dep,
)
plt.tight_layout(pad=0.5)
st.pyplot(fig_dep, use_container_width=True)
plt.close(fig_dep)

insight_box(
    f"Each dot is one student. The x-axis is their **{selected_feat.replace('_', ' ')}** value; "
    "the y-axis is the SHAP contribution. The colour shows a secondary interaction feature chosen automatically."
)