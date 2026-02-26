import pandas as pd
import numpy as np
import os

DATA_PATH = "data/StudentPerformanceFactors.csv"

def load_and_clean_data():
    """Loads dataset and handles missing values/outliers."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Handle Missing Values
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Simple Outlier Removal (IQR method) for target
    Q1 = df['Exam_Score'].quantile(0.25)
    Q3 = df['Exam_Score'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['Exam_Score'] < (Q1 - 1.5 * IQR)) | (df['Exam_Score'] > (Q3 + 1.5 * IQR)))]
    
    return df

def get_features(df):
    """Dynamically selects features available in the dataset."""
    base_features = ["Hours_Studied"]
    if "Sleep_Hours" in df.columns:
        base_features.append("Sleep_Hours")
    if "Attendance" in df.columns:
        base_features.append("Attendance")
    return base_features