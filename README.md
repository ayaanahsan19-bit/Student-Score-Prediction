🎓 Student Score Prediction System
An end-to-end Machine Learning application built for the Elevo Internship Program. This project predicts student exam scores based on study habits, attendance, and other factors using a production-grade project structure.

📁 Dataset
Source: Student Performance Factors (Kaggle)
The dataset contains information about students' study habits, sleep, attendance, and previous grades.
🛠️ Tools & Libraries
*   **Language:** Python
*   **Web Framework:** Streamlit
*   **Machine Learning:** Scikit-learn, XGBoost, Statsmodels
*   **Visualization:** Matplotlib, Seaborn
*   **Explainability:** SHAP
*   **Experiment Tracking:** MLflow

🚀 Key Features
Interactive Web App: Built with Streamlit for real-time predictions.
Multi-Model Architecture: Compares Linear Regression, Polynomial Regression, Huber Regressor, and XGBoost.
Model Explainability: Integrated SHAP values to interpret model decisions.
MLOps Structure: Professional folder structure separating training logic, utilities, and the application interface.

📁 Project Structure
This project follows industry best practices for maintainability:

student-score-prediction/
│
├── app.py # Main Streamlit application entry point
├── train_pipeline.py # Script to train models and save artifacts
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── pages/ # Streamlit multipage app pages
│ ├── 1_Dashboard.py # Interactive prediction dashboard
│ ├── 2_Analytics.py # Exploratory Data Analysis (EDA)
│ ├── 3_Model_Comparison.py # Model performance metrics & residuals
│ └── 4_Explainability.py # SHAP analysis page
│
├── utils/ # Helper functions
│ └── preprocessing.py # Data cleaning and feature engineering
│
├── models/ # Saved model artifacts (.pkl)
│ ├── xgb_model.pkl
│ ├── linear_model.pkl
│ └── metrics.json
│
└── data/ # Dataset location
└── StudentPerformanceFactors.csv

🚀 Project Workflow
Data Engineering:
Processed the dataset by handling missing values via median imputation and removing statistical outliers using the IQR method.
Model Training Pipeline:
Developed a modular pipeline to train and evaluate four distinct models: Linear Regression, Huber Regressor, Polynomial Regression, and XGBoost.
Deployment Architecture:
Serialized trained models and metrics to disk (.pkl/.json) to separate training logic from the inference application.
Interactive Dashboard:
Built a multipage Streamlit application enabling users to generate predictions, analyze data distributions, and compare model performance.
Model Explainability:
Integrated SHAP values to provide transparency into feature importance and individual prediction drivers.
