# 🎓 EduPredict AI — Student Score Prediction System

> An end-to-end Machine Learning web application built for the **Elevo Internship Program**.  
> Predicts student exam scores from study habits, attendance, and sleep using a fully production-grade, multi-page Streamlit app.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red?style=flat-square&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-6.6-3F4F75?style=flat-square&logo=plotly)
![SHAP](https://img.shields.io/badge/SHAP-0.50-blueviolet?style=flat-square)

---

## ✨ What's New (v2)

- **Full UI overhaul** — shared CSS theme (`utils/styles.py`) with Inter font, gradient banners, stat cards, and insight callouts applied across every page
- **Plotly-powered charts** — all visualisations replaced with fully interactive Plotly figures (hover, zoom, pan)
- **Streamlit theme config** — `.streamlit/config.toml` sets the indigo primary colour, clean white background
- **Home page redesigned** — hero banner, live stat cards (best R², model count, lowest MSE), feature navigation cards, tech-stack pills
- **Dashboard upgraded** — live gauge chart with colour zones & delta vs dataset mean, grade badge (A+→F), feature importance bar chart, context scatter with star marker
- **Analytics expanded** — KPI row, OLS trendline scatter, sleep distribution, score percentiles, full interactive heatmap, statistical summary table
- **Model Details enhanced** — styled scorecards with progress bars and ⚡ BEST badge, tabbed residual analysis for Linear vs XGBoost
- **Explainability deepened** — Plotly mean |SHAP| bar, beeswarm, per-student waterfall with profile card, feature dependence plot with dropdown

---

## 📸 App Pages

| Page | Description |
|---|---|
| 🏠 **Home** | Hero banner, live model stat cards, feature overview, tech stack |
| 📊 **Prediction Dashboard** | Live sliders → XGBoost prediction → gauge + grade badge + scatter |
| 📈 **Data Analytics** | Full EDA: distribution, correlation, scatter, percentiles, heatmap |
| ⚙️ **Model Comparison** | Scorecards, R²/MSE bar charts, tabbed residual diagnostics |
| 🧠 **Explainability** | SHAP global importance, beeswarm, waterfall, dependence plot |

---

## 🗂️ Project Structure

```
edupredict-ai/
│
├── app.py                          # Home page (hero + stat cards + navigation)
├── train_pipeline.py               # One-shot model training script
├── requirements.txt                # All Python dependencies
├── README.md
│
├── .streamlit/
│   └── config.toml                 # Streamlit theme (indigo, white, Inter)
│
├── pages/
│   ├── 1_📊_Dashboard.py           # Prediction dashboard (gauge, grade, scatter)
│   ├── 2_📈_Analytics.py           # Interactive EDA (Plotly)
│   ├── 3_⚙️_Model_Details.py       # Model comparison + residual analysis
│   └── 4_Explainability.py         # SHAP deep-dive
│
├── utils/
│   ├── preprocessing.py            # Data loading, cleaning, feature selection
│   └── styles.py                   # Shared CSS theme + helper render functions
│
├── models/                         # Saved artifacts (auto-generated)
│   ├── xgb_model.pkl
│   ├── linear_model.pkl
│   ├── huber_model.pkl
│   ├── poly_model.pkl
│   ├── features.pkl
│   └── metrics.json
│
└── data/
    └── StudentPerformanceFactors.csv
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Web Framework | Streamlit 1.54 |
| ML — Gradient Boosting | XGBoost 3.2 |
| ML — Classical Models | Scikit-learn |
| Interactive Charts | Plotly 6.6 |
| Static Charts / SHAP vis | Matplotlib |
| Explainability | SHAP 0.50 |
| Experiment Tracking | MLflow |
| Data Processing | Pandas, NumPy |

---

## 🚀 Quick Start

### 1 — Clone & install

```bash
git clone https://github.com/ayaanahsan19-bit/Student-Score-Prediction.git
cd Student-Score-Prediction
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 2 — Train the models (run once)

```bash
python train_pipeline.py
```

This saves four trained models and a `metrics.json` file to `models/`.

### 3 — Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 ML Pipeline

```
Raw CSV
  └─► IQR outlier removal + median imputation   (utils/preprocessing.py)
        └─► Feature selection (Hours_Studied, Sleep_Hours, Attendance)
              └─► Train / Test split (80 / 20, seed=42)
                    ├─► Linear Regression      → linear_model.pkl
                    ├─► Huber Regressor        → huber_model.pkl
                    ├─► Polynomial Reg (deg 2) → poly_model.pkl
                    └─► XGBoost Regressor      → xgb_model.pkl  ← best model
                          └─► metrics.json  (R², MSE for all four)
```

### Model Performance

| Model | R² Score | MSE |
|---|---|---|
| Linear Regression | ~0.723 | ~2.88 |
| Huber Regressor | ~0.720 | ~2.90 |
| Polynomial Reg | ~0.785 | ~2.23 |
| **XGBoost** | **~0.950+** | **~0.52** |

---

## 📊 Dataset

**Source:** [Student Performance Factors — Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

Key columns used:

| Feature | Description |
|---|---|
| `Hours_Studied` | Weekly study hours |
| `Sleep_Hours` | Average nightly sleep |
| `Attendance` | Class attendance rate (%) |
| `Exam_Score` | Target — final exam score (%) |

---

## 📁 Key Files Explained

| File | Role |
|---|---|
| `utils/styles.py` | Single source of truth for all CSS styling, injected on every page via `apply_theme()` |
| `.streamlit/config.toml` | Streamlit native theme override (primary colour, fonts, backgrounds) |
| `train_pipeline.py` | Trains all 4 models, saves `.pkl` artifacts and `metrics.json` |
| `pages/1_📊_Dashboard.py` | XGBoost prediction with Plotly gauge, grade badge, feature importance |
| `pages/4_Explainability.py` | SHAP beeswarm, waterfall per student, dependence plot |

---

## 👤 Author

**Ayaan Ahsan** — Elevo Internship Task  
Built with ❤️ using Streamlit · XGBoost · SHAP · Plotly
