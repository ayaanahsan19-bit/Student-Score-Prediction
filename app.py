import streamlit as st
import os

st.set_page_config(page_title="Student Score AI", page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Prediction System")
st.markdown("---")

st.markdown("""
### Welcome to the Internship Project Dashboard
This application demonstrates an end-to-end Machine Learning pipeline for predicting student exam scores.

**Key Features:**
*   **Multi-Model Comparison:** Linear, Polynomial, and XGBoost models.
*   **Explainability:** SHAP values for model interpretation.
*   **MLOps Ready:** Structured with production-grade code quality.

**Navigation:**
👉 Use the sidebar to explore:
1.  **Dashboard:** Interactive prediction tool.
2.  **Analytics:** Exploratory Data Analysis (EDA).
3.  **Model Comparison:** Performance metrics & Residual analysis.
4.  **Explainability:** SHAP deep-dive.
""")

# Check if models exist
if not os.path.exists("models/xgb_model.pkl"):
    st.warning("⚠️ Models not found! Please run `train_pipeline.py` first to generate the models.")
    st.code("python train_pipeline.py", language='bash')
else:
    st.success("✅ Models are trained and ready!")

st.markdown("<br><hr><center>Built with ❤️ by Ayaan Ahsan | Elevo Internship Task</center>", unsafe_allow_html=True)