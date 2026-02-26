import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
import mlflow.sklearn
import os

from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from xgboost import XGBRegressor
import statsmodels.api as sm

# =====================================================
# CONFIG
# =====================================================
RANDOM_STATE = 42
DATA_PATH = "data/StudentPerformanceFactors.csv"

# =====================================================
# LOAD DATA
# =====================================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Dataset not found.")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Dataset Loaded:", df.shape)

# =====================================================
# EDA + OUTLIER DETECTION
# =====================================================
numeric_cols = df.select_dtypes(include=np.number).columns

# IQR Outlier Detection
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
            (df[numeric_cols] > (Q3 + 1.5 * IQR)))

print("Total Outliers Detected:", outliers.sum().sum())

# Optional: remove extreme outliers
df_clean = df[~outliers.any(axis=1)]

# =====================================================
# FEATURES
# =====================================================
features = ["Hours_Studied"]

if "Sleep_Hours" in df.columns:
    features.append("Sleep_Hours")
if "Attendance" in df.columns:
    features.append("Attendance")

X = df_clean[features]
y = df_clean["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# =====================================================
# MLflow Tracking
# =====================================================
mlflow.set_experiment("Student_Score_Model_Comparison")

with mlflow.start_run():

    # -------------------------------------------------
    # 1. Linear Regression
    # -------------------------------------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    r2_lr = r2_score(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)

    mlflow.log_metric("LR_R2", r2_lr)
    mlflow.log_metric("LR_MSE", mse_lr)

    # -------------------------------------------------
    # 2. Robust Regression (Huber)
    # -------------------------------------------------
    huber = HuberRegressor()
    huber.fit(X_train, y_train)
    y_pred_huber = huber.predict(X_test)

    r2_huber = r2_score(y_test, y_pred_huber)
    mlflow.log_metric("Huber_R2", r2_huber)

    # -------------------------------------------------
    # 3. Polynomial + Scaling
    # -------------------------------------------------
    poly_model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2)),
        ("lr", LinearRegression())
    ])

    poly_model.fit(X_train, y_train)
    y_pred_poly = poly_model.predict(X_test)

    r2_poly = r2_score(y_test, y_pred_poly)
    mlflow.log_metric("Poly_R2", r2_poly)

    # -------------------------------------------------
    # 4. XGBoost
    # -------------------------------------------------
    xgb = XGBRegressor(random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)

    y_pred_xgb = xgb.predict(X_test)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    mlflow.log_metric("XGB_R2", r2_xgb)

    mlflow.sklearn.log_model(xgb, "xgboost_model")

# =====================================================
# CONFIDENCE INTERVALS
# =====================================================
X_const = sm.add_constant(X_train)
ols = sm.OLS(y_train, X_const).fit()

print("\nConfidence Intervals:")
print(ols.conf_int())

# =====================================================
# RESIDUAL DIAGNOSTICS
# =====================================================
residuals = y_test - y_pred_lr

plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(y_pred_lr, residuals)
plt.axhline(0, color='red')
plt.title("Residuals vs Predicted")
plt.show()

# Q-Q Plot
sm.qqplot(residuals, line='45')
plt.title("Q-Q Plot")
plt.show()

# =====================================================
# BIAS-VARIANCE ESTIMATION (Bootstrap)
# =====================================================
boot_preds = []

for _ in range(100):
    X_res, y_res = resample(X_train, y_train)
    model = LinearRegression()
    model.fit(X_res, y_res)
    boot_preds.append(model.predict(X_test))

boot_preds = np.array(boot_preds)

bias = np.mean((np.mean(boot_preds, axis=0) - y_test.values) ** 2)
variance = np.mean(np.var(boot_preds, axis=0))

print("\nBias:", bias)
print("Variance:", variance)

# =====================================================
# SHAP EXPLAINABILITY
# =====================================================
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)

# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n===== MODEL PERFORMANCE =====")
print(f"Linear R2: {r2_lr:.3f}")
print(f"Huber R2: {r2_huber:.3f}")
print(f"Polynomial R2: {r2_poly:.3f}")
print(f"XGBoost R2: {r2_xgb:.3f}")