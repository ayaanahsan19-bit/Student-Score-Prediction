import pandas as pd
import numpy as np
import json
import joblib
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from utils.preprocessing import load_and_clean_data, get_features

# Setup directories
os.makedirs("models", exist_ok=True)

# 1. Load Data
df = load_and_clean_data()
features = get_features(df)

X = df[features]
y = df['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store metrics
metrics = {}

# ---------------------------------------------------------
# Model 1: Linear Regression
# ---------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
metrics['Linear_Regression'] = {'r2': r2_score(y_test, pred_lr), 'mse': mean_squared_error(y_test, pred_lr)}
joblib.dump(lr, "models/linear_model.pkl")

# ---------------------------------------------------------
# Model 2: Robust Regression (Huber)
# ---------------------------------------------------------
huber = HuberRegressor()
huber.fit(X_train, y_train)
pred_huber = huber.predict(X_test)
metrics['Huber_Regressor'] = {'r2': r2_score(y_test, pred_huber), 'mse': mean_squared_error(y_test, pred_huber)}
joblib.dump(huber, "models/huber_model.pkl")

# ---------------------------------------------------------
# Model 3: Polynomial Regression (Degree 2)
# ---------------------------------------------------------
poly_model = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2)),
    ("lr", LinearRegression())
])
poly_model.fit(X_train, y_train)
pred_poly = poly_model.predict(X_test)
metrics['Polynomial_Reg'] = {'r2': r2_score(y_test, pred_poly), 'mse': mean_squared_error(y_test, pred_poly)}
joblib.dump(poly_model, "models/poly_model.pkl")

# ---------------------------------------------------------
# Model 4: XGBoost (State of the Art)
# ---------------------------------------------------------
xgb = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
metrics['XGBoost'] = {'r2': r2_score(y_test, pred_xgb), 'mse': mean_squared_error(y_test, pred_xgb)}
joblib.dump(xgb, "models/xgb_model.pkl")

# Save features used
joblib.dump(features, "models/features.pkl")

# ---------------------------------------------------------
# Save Metrics to JSON
# ---------------------------------------------------------
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Training Complete. Models saved to /models")
print("Metrics:", metrics)