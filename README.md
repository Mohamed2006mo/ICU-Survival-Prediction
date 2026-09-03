# 🏥 ICU Mortality Risk Predictor

A machine learning pipeline that predicts in-hospital mortality risk for ICU patients, built on 91,713 real ICU encounters. The project covers the full lifecycle — data cleaning, feature engineering, handling severe class imbalance, model comparison, threshold tuning, and deployment as an interactive web app.

**🔗 Live App:** [icu-survival-prediction-c9a8bq2znezjsshra7dxcg.streamlit.app](https://icu-survival-prediction-c9a8bq2znezjsshra7dxcg.streamlit.app)

---

## 📊 Project Overview

| | |
|---|---|
| **Dataset** | 91,713 ICU encounters, 230+ raw clinical features |
| **Target** | In-hospital mortality (severe imbalance: 91.3% survived vs. 8.6% died) |
| **Final Model** | XGBoost (AUROC 0.893, AUPRC 0.554) |
| **Selected Features** | 40, via Mutual Information + Recursive Feature Elimination |
| **Decision Threshold** | 0.40 (tuned for recall on the minority class) |
| **Deployment** | Streamlit multi-page web app |

## 🧠 What This Project Covers

- **Data Cleaning** — missingness audit, dropped 74 columns with >50% missing values, median/most-frequent imputation, one-hot encoding
- **Feature Engineering** — shock index, GCS total score, vital-sign ranges, SpO2/respiratory-rate ratio, log-transformed lab values
- **Class Imbalance** — compared SMOTE against XGBoost's native `scale_pos_weight`; moved away from SMOTE after it produced unrealistic synthetic patient records and capped recall
- **Feature Selection** — two-stage Mutual Information → Recursive Feature Elimination, narrowing 156 encoded features to the 40 most informative
- **Model Comparison** — Logistic Regression, Random Forest, and XGBoost trained and evaluated on identical data; XGBoost won on both AUROC and AUPRC
- **Threshold Tuning** — swept the precision-recall trade-off and selected a threshold based on the clinical cost of a missed death vs. a false alarm
- **Validation** — 5-fold stratified cross-validation confirmed the result wasn't a lucky train/test split (AUROC ±0.0012, AUPRC ±0.0056)
- **Deployment** — a 5-page Streamlit app (Overview, Data Exploration, Prediction, Model Performance, Model Comparison) that replicates the full preprocessing pipeline for live predictions on raw clinical input

## 📁 Repository Structure

```
ICU-Survival-Prediction/
├── App/
│   ├── app.py                    # Streamlit application
│   ├── xgb_icu_model.pkl         # Trained XGBoost model
│   ├── num_imputer.pkl           # Fitted numeric imputer (medians)
│   ├── cat_imputer.pkl           # Fitted categorical imputer
│   ├── final_40_features.json    # Selected feature list & order
│   ├── chosen_threshold.json     # Tuned decision threshold
│   ├── encoded_columns.json      # Full one-hot column reference
│   └── requirements.txt
├── ICU_Predictor_annotated.ipynb # Full training notebook (EDA → modeling → evaluation)
└── README.md
```

## 🚀 Running Locally

```bash
git clone https://github.com/Mohamed2006mo/ICU-Survival-Prediction.git
cd ICU-Survival-Prediction/App
pip install -r requirements.txt
streamlit run app.py
```

## 🖥️ App Pages

- **Overview** — project summary and key stats
- **Data Exploration** — class imbalance, mortality-by-age trends, top features by mutual information
- **Prediction** — enter raw clinical values (APACHE scores, GCS, vitals, labs) and get a live mortality risk estimate
- **Model Performance** — XGBoost's confusion matrix and cross-validation stability
- **All Models Comparison** — reported AUROC/AUPRC across Logistic Regression, Random Forest, and XGBoost

## 📈 Results

| Model | AUROC | AUPRC |
|---|---|---|
| Logistic Regression | 0.874 | 0.462 |
| Random Forest | 0.889 | 0.511 |
| **XGBoost (deployed)** | **0.893** | **0.554** |

At the chosen threshold (0.40), the deployed model catches 84% of actual deaths (1,327 of 1,583) in the held-out test set.

## ⚠️ Limitations

- Feature selection (MI/RFE) was performed once on the original train/test split, not re-run inside each cross-validation fold
- Single-dataset source — generalization to other hospital systems is untested
- The 0.40 threshold reflects one reasonable clinical trade-off, not a universally "correct" cutoff
- This tool is a decision-support estimate, not a medical diagnosis

## 🛠️ Tech Stack

Python · pandas · scikit-learn · XGBoost · Streamlit · Plotly

## 👤 Author

**Mohamed** — Data Science, Al-Shorouk Academy
