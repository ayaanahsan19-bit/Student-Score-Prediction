import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import os

# ---------------------------------------------------------
# STEP 1: Load the Dataset
# ---------------------------------------------------------
# The code looks for the CSV file inside the 'data' folder.
file_path = 'data/StudentPerformanceFactors.csv'

# Safety check: if file not found, stop the code and tell user.
if not os.path.exists(file_path):
    print(f"Error: File not found at '{file_path}'.")
    print("Please make sure you have extracted the ZIP file and placed the CSV inside the 'data' folder.")
    exit()

# Load the data
df = pd.read_csv(file_path)
print("Dataset loaded successfully!\n")
print("First 5 rows of the dataset:")
print(df.head())
print("\n" + "-"*50 + "\n")

# ---------------------------------------------------------
# STEP 2: Data Cleaning
# ---------------------------------------------------------
# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Check for missing values in the columns we need
print("Checking for missing values...")
print(df[['Hours_Studied', 'Exam_Score']].isnull().sum())

# Fill missing values with median (standard practice for numeric data)
df['Hours_Studied'] = df['Hours_Studied'].fillna(df['Hours_Studied'].median())
df['Exam_Score'] = df['Exam_Score'].fillna(df['Exam_Score'].median())
print("Missing values handled.\n")

# ---------------------------------------------------------
# STEP 3: Basic Visualization
# ---------------------------------------------------------
print("Generating Visualization 1: Study Hours vs Exam Score...")
plt.figure(figsize=(10, 6))
plt.scatter(df['Hours_Studied'], df['Exam_Score'], color='blue', alpha=0.5)
plt.title('Study Hours vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.grid(True)
plt.savefig('1_visualization_study_hours.png')
plt.show()

# ---------------------------------------------------------
# STEP 4: Split Dataset (Training and Testing)
# ---------------------------------------------------------
# Feature (X) needs to be a 2D array, Target (y) is 1D
X = df[['Hours_Studied']]
y = df['Exam_Score']

# Split: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}\n")

# ---------------------------------------------------------
# STEP 5: Train Linear Regression Model
# ---------------------------------------------------------
print("Training Linear Regression Model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on test set
y_pred_lr = lr_model.predict(X_test)

# Evaluation Metrics
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\n--- Linear Regression Performance ---")
print(f"Mean Squared Error (MSE): {mse_lr:.2f}")
print(f"R2 Score: {r2_lr:.2f}")
print(f"Model Equation: Score = {lr_model.coef_[0]:.2f} * Hours + {lr_model.intercept_:.2f}")

# Visualize Linear Regression Predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Scores')
plt.plot(X_test, y_pred_lr, color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.savefig('2_linear_regression_result.png')
plt.show()

# ---------------------------------------------------------
# BONUS 1: Polynomial Regression
# ---------------------------------------------------------
print("\n--- Bonus: Training Polynomial Regression (Degree 2) ---")
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Polynomial Regression MSE: {mse_poly:.2f}")
print(f"Polynomial Regression R2 Score: {r2_poly:.2f}")

# Visualize Polynomial Regression
# To plot a smooth curve, we sort the X values
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = poly_features.transform(X_range)
y_range_pred = poly_model.predict(X_range_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Scores')
plt.plot(X_range, y_range_pred, color='green', linewidth=2, label='Polynomial Curve')
plt.title('Polynomial Regression: Actual vs Predicted')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.savefig('3_polynomial_regression_result.png')
plt.show()

# ---------------------------------------------------------
# BONUS 2: Experimenting with different features
# ---------------------------------------------------------
print("\n--- Bonus: Experimenting with Multiple Features ---")
# We try to use 'Sleep_Hours' and 'Attendance' if available
features_to_try = ['Hours_Studied', 'Sleep_Hours', 'Attendance']
available_features = [f for f in features_to_try if f in df.columns]

if len(available_features) > 1:
    print(f"Features used for multi-variable model: {available_features}")
    
    # Prepare data
    X_multi = df[available_features].copy()
    # Fill missing values for all selected columns
    for col in available_features:
        X_multi[col] = X_multi[col].fillna(X_multi[col].median())
    
    y_multi = df['Exam_Score']

    # Split
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
    
    # Train
    multi_model = LinearRegression()
    multi_model.fit(X_train_m, y_train_m)
    y_pred_multi = multi_model.predict(X_test_m)
    
    # Evaluate
    r2_multi = r2_score(y_test_m, y_pred_multi)
    
    print(f"R2 Score (Multiple Features): {r2_multi:.2f}")
    print(f"R2 Score (Single Feature): {r2_lr:.2f}")
    print("Observation: Adding more features usually improves the R2 Score.")
else:
    print("Not enough features available in the dataset to perform multi-feature comparison.")