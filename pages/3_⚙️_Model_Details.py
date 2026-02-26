import streamlit as st
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import load_and_clean_data
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Model Comparison", layout="wide")

st.title("⚙️ Model Performance & Comparison")

# Load Metrics
try:
    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)
except:
    st.error("Run train_pipeline.py first.")
    st.stop()

# 1. Metrics Table
st.subheader("Performance Metrics")
df_metrics = pd.DataFrame(metrics).T
df_metrics.columns = ["R2 Score", "MSE"]
st.dataframe(df_metrics.style.background_gradient(cmap='Greens'), use_container_width=True)

# 2. Residual Analysis (Recalculating on the fly for visualization)
df = load_and_clean_data()
features = joblib.load("models/features.pkl")
X = df[features]
y = df['Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = joblib.load("models/linear_model.pkl")
y_pred = lr_model.predict(X_test)
residuals = y_test - y_pred

col1, col2 = st.columns(2)

with col1:
    st.subheader("Residual Distribution (Linear Model)")
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, color='purple', ax=ax)
    ax.set_title("Residuals should be normally distributed")
    st.pyplot(fig)

with col2:
    st.subheader("Predicted vs Actual")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)