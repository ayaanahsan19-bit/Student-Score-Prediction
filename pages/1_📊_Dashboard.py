import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import load_and_clean_data

st.set_page_config(page_title="Prediction Dashboard", layout="wide")

# Load Model and Data
try:
    model = joblib.load("models/xgb_model.pkl") # Using best model
    features = joblib.load("models/features.pkl")
    df = load_and_clean_data()
except:
    st.error("Models not found. Please run train_pipeline.py")
    st.stop()

st.title("📊 Prediction Dashboard")

# Sidebar Inputs
with st.sidebar:
    st.header("Student Inputs")
    input_data = {}
    for feat in features:
        if feat == "Hours_Studied":
            val = st.slider(feat, 0, 50, 10)
        elif feat == "Attendance":
            val = st.slider(feat, 0, 100, 75)
        else:
            val = st.slider(feat, 0, 15, 7)
        input_data[feat] = [val]

input_df = pd.DataFrame(input_data)

# Prediction
prediction = model.predict(input_df)[0]

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Predicted Exam Score", f"{prediction:.1f}%")
    st.info(f"Model used: **XGBoost**")
    
    # Feature Importance (Static display for XGB)
    st.subheader("Feature Impact")
    fig_imp, ax_imp = plt.subplots(figsize=(5, 3))
    sns.barplot(x=model.feature_importances_, y=features, ax=ax_imp, palette="viridis")
    st.pyplot(fig_imp)

with col2:
    st.subheader("Context Visualization")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='Hours_Studied', y='Exam_Score', alpha=0.3, color='gray', ax=ax)
    ax.scatter(input_df['Hours_Studied'], prediction, color='red', s=200, label='Prediction', edgecolors='black', zorder=5)
    ax.legend()
    ax.set_title("Prediction vs Historical Data")
    st.pyplot(fig)