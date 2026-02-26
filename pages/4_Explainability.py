import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from utils.preprocessing import load_and_clean_data

st.set_page_config(page_title="Explainability", layout="wide")

st.title("🧠 Model Explainability (SHAP)")

st.markdown("""
**Why did the model make this prediction?**
We use SHAP (SHapley Additive exPlanations) to understand feature importance.
""")

# Load data and model
try:
    df = load_and_clean_data()
    features = joblib.load("models/features.pkl")
    model = joblib.load("models/xgb_model.pkl") # XGBoost works best with TreeExplainer
    X = df[features].sample(100) # Sample for speed
except:
    st.error("Models not found. Run train_pipeline.py")
    st.stop()

# Calculate SHAP values
with st.spinner("Calculating SHAP values..."):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

# Plot 1: Summary Plot
st.subheader("Global Feature Importance")
st.caption("This shows which features are most important globally across all students.")
fig1, ax1 = plt.subplots()
shap.summary_plot(shap_values, X, show=False)
st.pyplot(fig1)

# Plot 2: Waterfall plot for a single prediction
st.subheader("Single Prediction Explanation")
index_to_explain = st.slider("Select Student Index", 0, len(X)-1, 0)

fig2, ax2 = plt.subplots()
shap.waterfall_plot(shap_values[index_to_explain], show=False)
st.pyplot(fig2)