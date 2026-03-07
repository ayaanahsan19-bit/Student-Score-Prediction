import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from utils.preprocessing import load_and_clean_data
from utils.styles import apply_theme, page_header, section_title, insight_box

st.set_page_config(page_title="Prediction Dashboard", page_icon="📊", layout="wide")
apply_theme()
page_header("📊 Prediction Dashboard",
            "Adjust the sliders and get a real-time exam score prediction powered by XGBoost")

# ── Load model & data ─────────────────────────────────────────────────────────
try:
    model    = joblib.load("models/xgb_model.pkl")
    features = joblib.load("models/features.pkl")
    df       = load_and_clean_data()
except Exception as e:
    st.error(f"⚠️ Could not load models: {e}. Please run `python train_pipeline.py` first.")
    st.stop()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Student Parameters")
    st.markdown("---")
    _slider_cfg = {
        "Hours_Studied": ("📚 Hours Studied",   0, 50,  10, "hours / week"),
        "Attendance":    ("🏫 Attendance Rate",  0, 100, 75, "%"),
        "Sleep_Hours":   ("😴 Sleep Hours",      0, 15,   7, "hrs / night"),
    }
    input_data = {}
    for feat in features:
        if feat in _slider_cfg:
            lbl, mn, mx, default, unit = _slider_cfg[feat]
            val = st.slider(lbl, mn, mx, default, help=f"Measured in {unit}")
        else:
            val = st.slider(feat.replace("_", " "), 0, 100, 50)
        input_data[feat] = [val]

    st.markdown("---")
    st.caption("🤖 Model: XGBoost Regressor")
    st.caption("📊 Dataset: StudentPerformanceFactors")

input_df  = pd.DataFrame(input_data)
raw_pred  = float(model.predict(input_df)[0])
prediction = max(0.0, min(100.0, raw_pred))

# ── Grade helper ──────────────────────────────────────────────────────────────
def get_grade(score):
    if score >= 90: return "A+", "Excellent",      "#10b981", "score-excellent"
    if score >= 80: return "A",  "Very Good",       "#3b82f6", "score-good"
    if score >= 70: return "B",  "Good",            "#6366f1", "score-good"
    if score >= 60: return "C",  "Average",         "#f59e0b", "score-average"
    if score >= 50: return "D",  "Below Average",   "#ef4444", "score-poor"
    return             "F",  "Needs Improvement","#dc2626", "score-poor"

grade, grade_label, grade_color, badge_class = get_grade(prediction)
avg_score = float(df["Exam_Score"].mean())

# ── Layout: result | gauge | importance ──────────────────────────────────────
col1, col2, col3 = st.columns([1, 1.6, 1.4])

with col1:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    section_title("🎯 Result")
    st.markdown(
        f'<div class="{badge_class} score-badge">{prediction:.1f}%</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="text-align:center;margin-top:0.7rem;">'
        f'<span style="font-size:1.4rem;font-weight:700;color:{grade_color};">Grade {grade}</span><br>'
        f'<span style="color:#64748b;font-size:0.88rem;">{grade_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    section_title("📌 Inputs")
    for k, v in input_data.items():
        st.markdown(f"**{k.replace('_',' ')}:** {v[0]}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    section_title("⚡ Score Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        delta={
            "reference": avg_score,
            "increasing": {"color": "#10b981"},
            "decreasing": {"color": "#ef4444"},
            "suffix": "%",
        },
        number={"suffix": "%", "font": {"size": 42, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar":  {"color": grade_color, "thickness": 0.32},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  50], "color": "#fee2e2"},
                {"range": [50, 70], "color": "#fef9c3"},
                {"range": [70, 85], "color": "#d1fae5"},
                {"range": [85,100], "color": "#a7f3d0"},
            ],
            "threshold": {
                "line": {"color": "#475569", "width": 3},
                "thickness": 0.75,
                "value": avg_score,
            },
        },
    ))
    fig_gauge.update_layout(
        height=310, margin=dict(l=20, r=20, t=40, b=20),
        font={"family": "Inter"}, paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    insight_box(
        f"Dataset average: **{avg_score:.1f}%**. "
        f"This prediction is **{'above' if prediction >= avg_score else 'below'}** average "
        f"by **{abs(prediction - avg_score):.1f}** points."
    )

with col3:
    section_title("📊 Feature Weights")
    importances  = model.feature_importances_
    feat_labels  = [f.replace("_", " ") for f in features]
    palette      = ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"]
    sorted_pairs = sorted(zip(importances, feat_labels), reverse=True)
    s_vals, s_lbls = zip(*sorted_pairs)

    fig_imp = go.Figure(go.Bar(
        x=list(s_vals), y=list(s_lbls),
        orientation="h",
        marker=dict(color=palette[:len(features)]),
        text=[f"{v:.1%}" for v in s_vals],
        textposition="outside",
    ))
    fig_imp.update_layout(
        height=220,
        margin=dict(l=10, r=50, t=10, b=10),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "size": 13},
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Quick performance tips
    st.markdown('<div class="info-card" style="margin-top:0.5rem;">', unsafe_allow_html=True)
    section_title("💡 Tips to Improve")
    tips = {
        "Hours_Studied": ("📚", "Study more consistently — even 1 extra hour helps."),
        "Attendance":    ("🏫", "Higher attendance correlates strongly with better scores."),
        "Sleep_Hours":   ("😴", "7–9 hours of sleep improves retention and focus."),
    }
    for feat in features:
        if feat in tips:
            icon, tip = tips[feat]
            st.markdown(f"{icon} {tip}")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Context scatter ───────────────────────────────────────────────────────────
st.markdown("---")
section_title("🗺️ Your Prediction vs Historical Data")

color_col = "Attendance" if "Attendance" in df.columns else None
fig_scatter = px.scatter(
    df, x="Hours_Studied", y="Exam_Score",
    color=color_col,
    color_continuous_scale="Viridis",
    opacity=0.35,
    labels={"Hours_Studied": "Hours Studied / Week", "Exam_Score": "Exam Score (%)"},
)
fig_scatter.add_trace(go.Scatter(
    x=input_df["Hours_Studied"],
    y=[prediction],
    mode="markers",
    marker=dict(size=20, color="#ef4444", symbol="star",
                line=dict(width=2, color="white")),
    name="Your Prediction",
))
fig_scatter.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,252,1)",
    font={"family": "Inter"},
    height=380,
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
    yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
)
st.plotly_chart(fig_scatter, use_container_width=True)