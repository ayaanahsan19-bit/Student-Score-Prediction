import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import load_and_clean_data

st.set_page_config(page_title="Analytics", layout="wide")

df = load_and_clean_data()

st.title("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Exam_Score'], kde=True, color='teal', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Correlation Matrix")
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr[['Exam_Score']].sort_values(by='Exam_Score', ascending=False), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

st.subheader("Hours Studied vs Score by Attendance")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='Hours_Studied', y='Exam_Score', hue='Attendance', palette='viridis', ax=ax2)
st.pyplot(fig2)