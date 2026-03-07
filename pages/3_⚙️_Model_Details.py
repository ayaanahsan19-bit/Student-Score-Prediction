import streamlit as st
import pandas as pd
import json
import joblib
import plotly.graph_objects as go
import plotly.express as px
from utils.preprocessing import load_and_clean_data
from sklearn.model_selection import train_test_split
from utils.styles import apply_theme, page_header, section_title, insight_box

st.set_page_config(page_title="Model Comparison", page_icon="⚙️", layout="wide")
apply_theme()
page_header("⚙️ Model Performance & Comparison",
            "Evaluate all trained models side-by-side with residual diagnostics")

# ── Load metrics ──────────────────────────────────────────────────────────────
try:
    with open("models/metrics.json") as f:
        metrics = json.load(f)
except Exception:
    st.error("Run `python train_pipeline.py` first.")
    st.stop()

try:
    df      = load_and_clean_data()
    features = joblib.load("models/features.pkl")
    X, y     = df[features], df["Exam_Score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    lr_model  = joblib.load("models/linear_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    y_pred_lr  = lr_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

best_key = max(metrics, key=lambda k: metrics[k]["r2"])

# ── Model scorecards ──────────────────────────────────────────────────────────
section_title("🏆 Model Scorecards")

_icons = {
    "Linear_Regression": "📏",
    "Huber_Regressor":   "🔒",
    "Polynomial_Reg":    "🌀",
    "XGBoost":           "⚡",
}
_descs = {
    "Linear_Regression": "Simple linear mapping between features and the target score",
    "Huber_Regressor":   "Robust regression that down-weights outlier influence",
    "Polynomial_Reg":    "Captures non-linear relationships (degree-2 polynomial)",
    "XGBoost":           "Gradient-boosted ensemble — highest accuracy",
}

cols = st.columns(len(metrics))
for col, (name, vals) in zip(cols, metrics.items()):
    r2, mse  = vals["r2"], vals["mse"]
    is_best  = name == best_key
    icon     = _icons.get(name, "🤖")
    desc     = _descs.get(name, "")
    border   = "#10b981" if is_best else "#e2e8f0"
    badge    = ('<span style="background:#10b981;color:white;border-radius:50px;'
                'padding:.1rem .5rem;font-size:.68rem;font-weight:700;">BEST</span> '
                if is_best else "")
    color    = "#10b981" if is_best else "#6366f1"
    bar_bg   = "linear-gradient(to right,#10b981,#059669)" if is_best else \
               "linear-gradient(to right,#6366f1,#8b5cf6)"

    with col:
        st.markdown(
            f'<div class="info-card" style="border-left:4px solid {border};text-align:center;">'
            f'<div style="font-size:2.2rem;">{icon}</div>'
            f'<div style="font-weight:600;font-size:.88rem;color:#1e293b;margin:.2rem 0;">'
            f'{badge}{name.replace("_"," ")}</div>'
            f'<div style="font-size:.73rem;color:#64748b;margin-bottom:.8rem;">{desc}</div>'
            f'<div style="font-size:1.65rem;font-weight:700;color:{color};">{r2:.1%}</div>'
            f'<div style="font-size:.68rem;color:#94a3b8;text-transform:uppercase;'
            f'letter-spacing:.06em;">R² Score</div>'
            f'<div style="height:6px;background:#e2e8f0;border-radius:3px;margin:.4rem 0;">'
            f'<div style="height:6px;width:{r2*100:.1f}%;background:{bar_bg};border-radius:3px;"></div>'
            f'</div>'
            f'<div style="font-size:.8rem;color:#64748b;">MSE: {mse:.4f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── R² vs MSE bar charts ──────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

names    = [n.replace("_", " ") for n in metrics]
r2_vals  = [v["r2"]  for v in metrics.values()]
mse_vals = [v["mse"] for v in metrics.values()]
bar_colors_r2  = ["#10b981" if n.replace(" ", "_") == best_key else "#6366f1" for n in names]
best_mse_idx   = mse_vals.index(min(mse_vals))
bar_colors_mse = ["#10b981" if i == best_mse_idx else "#f59e0b" for i in range(len(names))]

with col_a:
    section_title("📊 R² Score Comparison")
    fig_r2 = go.Figure(go.Bar(
        x=names, y=r2_vals, marker=dict(color=bar_colors_r2),
        text=[f"{v:.3f}" for v in r2_vals], textposition="outside",
    ))
    fig_r2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
        font={"family": "Inter"}, height=300,
        yaxis=dict(range=[0, 1.1], title="R² Score", gridcolor="#e2e8f0"),
        xaxis_title="Model",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_r2, use_container_width=True)

with col_b:
    section_title("📉 MSE Comparison (lower = better)")
    fig_mse = go.Figure(go.Bar(
        x=names, y=mse_vals, marker=dict(color=bar_colors_mse),
        text=[f"{v:.4f}" for v in mse_vals], textposition="outside",
    ))
    fig_mse.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
        font={"family": "Inter"}, height=300,
        yaxis=dict(title="Mean Squared Error", gridcolor="#e2e8f0"),
        xaxis_title="Model",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_mse, use_container_width=True)

st.markdown("---")

# ── Residual analysis with tabs ───────────────────────────────────────────────
section_title("🔬 Residual Analysis")

residuals_lr  = y_test - y_pred_lr
residuals_xgb = y_test - y_pred_xgb
min_val, max_val = float(y_test.min()), float(y_test.max())

tab1, tab2 = st.tabs(["📏 Linear Regression", "⚡ XGBoost"])

def _residual_charts(residuals, y_pred, color, title_suffix):
    c1, c2 = st.columns(2)
    with c1:
        fig_r = px.histogram(
            x=residuals, nbins=30, color_discrete_sequence=[color],
            labels={"x": "Residual"}, title=f"Residual Distribution — {title_suffix}",
        )
        fig_r.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
            font={"family": "Inter"}, height=310, showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_r, use_container_width=True)
    with c2:
        fig_p = px.scatter(
            x=y_test, y=y_pred, opacity=0.55,
            color_discrete_sequence=[color],
            labels={"x": "Actual Score", "y": "Predicted Score"},
            title=f"Predicted vs Actual — {title_suffix}",
        )
        fig_p.add_shape(
            type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
            line=dict(color="#ef4444", width=2, dash="dash"),
        )
        fig_p.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,1)",
            font={"family": "Inter"}, height=310,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_p, use_container_width=True)

with tab1:
    _residual_charts(residuals_lr,  y_pred_lr,  "#8b5cf6", "Linear Regression")
    insight_box("Residuals should be centred on 0 and normally distributed for a well-calibrated model.")

with tab2:
    _residual_charts(residuals_xgb, y_pred_xgb, "#10b981", "XGBoost")
    insight_box("XGBoost residuals cluster tighter around 0, confirming its superior fit on this dataset.")

st.markdown("---")

# ── Full metrics table ────────────────────────────────────────────────────────
section_title("📋 Full Metrics Table")
df_metrics = pd.DataFrame(metrics).T
df_metrics.columns = ["R² Score", "MSE"]
df_metrics["RMSE"]  = df_metrics["MSE"] ** 0.5
df_metrics["Grade"] = df_metrics["R² Score"].apply(
    lambda x: "🥇 Excellent" if x >= 0.9 else ("🥈 Good" if x >= 0.7 else "🥉 Fair")
)
st.dataframe(
    df_metrics.style
              .background_gradient(cmap="Greens", subset=["R² Score"])
              .background_gradient(cmap="Reds_r", subset=["MSE"])
              .format({"R² Score": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}"}),
    use_container_width=True,
)