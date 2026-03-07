import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.preprocessing import load_and_clean_data
from utils.styles import apply_theme, page_header, section_title, insight_box, sidebar_nav

st.set_page_config(page_title="Data Analytics", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
apply_theme()
sidebar_nav("Data Analytics")
page_header("📈 Exploratory Data Analysis",
            "Deep dive into the Student Performance dataset")

df = load_and_clean_data()
numeric_df = df.select_dtypes(include=["float64", "int64"])

# ── Top-level KPIs ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Students",  f"{len(df):,}")
c2.metric("Average Score",   f"{df['Exam_Score'].mean():.1f}%")
c3.metric("Std Deviation",   f"{df['Exam_Score'].std():.2f}")
c4.metric("Top Score",       f"{df['Exam_Score'].max():.0f}%")

st.markdown("---")

# ── Row 1: Distribution + Correlation ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    section_title("📊 Score Distribution")
    fig_dist = px.histogram(
        df, x="Exam_Score", nbins=30,
        color_discrete_sequence=["#6366f1"],
        marginal="rug",
        labels={"Exam_Score": "Exam Score (%)"},
    )
    fig_dist.update_traces(opacity=0.82)
    fig_dist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
        font={"family": "Inter"}, height=340, showlegend=False, bargap=0.06,
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    insight_box(
        f"Scores range from **{df['Exam_Score'].min():.0f}** to **{df['Exam_Score'].max():.0f}**. "
        f"The distribution is roughly bell-shaped around **{df['Exam_Score'].mean():.1f}%**."
    )

with col2:
    section_title("🔥 Feature Correlation with Exam Score")
    corr = (
        numeric_df.corr()[["Exam_Score"]]
        .drop("Exam_Score")
        .sort_values("Exam_Score", ascending=True)
    )
    colors = ["#ef4444" if v < 0 else "#10b981" for v in corr["Exam_Score"]]
    fig_corr = go.Figure(go.Bar(
        x=corr["Exam_Score"],
        y=[c.replace("_", " ") for c in corr.index],
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.3f}" for v in corr["Exam_Score"]],
        textposition="outside",
    ))
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
        font={"family": "Inter"}, height=340,
        xaxis=dict(title="Pearson Correlation", zeroline=True,
                   zerolinecolor="#94a3b8", gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=55, t=20, b=0),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# ── Row 2: Scatter + Sleep distribution ──────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    section_title("⏰ Hours Studied vs Exam Score")
    color_col = "Attendance" if "Attendance" in df.columns else None
    fig_scatter = px.scatter(
        df, x="Hours_Studied", y="Exam_Score",
        color=color_col,
        color_continuous_scale="Viridis",
        trendline="ols",
        opacity=0.45,
        labels={"Hours_Studied": "Hours Studied / Week", "Exam_Score": "Exam Score (%)"},
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
        font={"family": "Inter"}, height=330,
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col4:
    if "Sleep_Hours" in df.columns:
        section_title("😴 Sleep Hours Distribution")
        fig_sleep = px.histogram(
            df, x="Sleep_Hours",
            color_discrete_sequence=["#8b5cf6"],
            nbins=15, marginal="box",
            labels={"Sleep_Hours": "Sleep Hours per Night"},
        )
        fig_sleep.update_traces(opacity=0.82)
        fig_sleep.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
            font={"family": "Inter"}, height=330, showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_sleep, use_container_width=True)
    else:
        section_title("ℹ️ Sleep Data")
        st.info("Sleep_Hours feature not found in this dataset.")

st.markdown("---")

# ── Row 3: Score percentiles + Full heatmap ───────────────────────────────────
col5, col6 = st.columns(2)

with col5:
    section_title("📦 Score Percentiles")
    pcts   = [10, 25, 50, 75, 90, 95]
    pct_vals = [float(np.percentile(df["Exam_Score"], p)) for p in pcts]
    fig_pct = go.Figure(go.Bar(
        x=[f"P{p}" for p in pcts], y=pct_vals,
        marker=dict(
            color=pct_vals,
            colorscale=[[0, "#fee2e2"], [0.5, "#fef9c3"], [1, "#a7f3d0"]],
            showscale=False,
        ),
        text=[f"{v:.1f}%" for v in pct_vals],
        textposition="outside",
    ))
    fig_pct.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
        font={"family": "Inter"}, height=300,
        yaxis=dict(range=[0, 115], title="Score (%)", gridcolor="#e2e8f0"),
        xaxis_title="Percentile",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_pct, use_container_width=True)

with col6:
    section_title("🌡️ Full Correlation Heatmap")
    corr_full = numeric_df.corr()
    fig_heat = go.Figure(go.Heatmap(
        z=corr_full.values,
        x=[c.replace("_", " ") for c in corr_full.columns],
        y=[c.replace("_", " ") for c in corr_full.index],
        colorscale="RdBu", zmid=0,
        text=corr_full.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 11},
        hoverongaps=False,
    ))
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"}, height=300,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ── Statistical summary table ─────────────────────────────────────────────────
section_title("📋 Statistical Summary")
summary = numeric_df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
summary.columns = ["Mean", "Std Dev", "Min", "Q1", "Median", "Q3", "Max"]
st.dataframe(
    summary.style
           .background_gradient(cmap="Blues", subset=["Mean"])
           .format(precision=2),
    use_container_width=True,
)