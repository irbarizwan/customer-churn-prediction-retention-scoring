# DS-3: Customer Churn Prediction & Retention Scoring System
**Teyzix Core Internship | Data Science**

## Quick Start

```bash
# 1. Install dependencies
pip install scikit-learn xgboost lightgbm shap pandas numpy matplotlib seaborn streamlit

# 2. Run the full ML pipeline
python pipeline.py

# 3. Launch the dashboard
streamlit run dashboard/app.py
```

## What the pipeline does

1. Loads and cleans the Telco Customer Churn dataset
2. Engineers 16 behavioral and billing features
3. Trains and compares Logistic Regression, XGBoost, and LightGBM
4. Computes SHAP values for every prediction
5. Scores all customers weekly, assigns HIGH / MEDIUM / LOW risk
6. Generates per-customer intervention recommendations
7. Stores results in SQLite (drop-in replacement for PostgreSQL)
8. Exports a sample HIGH-risk call list

## Key outputs

| File | Description |
|---|---|
| `outputs/weekly_scores_YYYY-MM-DD.csv` | All customers with churn probability, risk, recommendations |
| `outputs/sample_call_list.csv` | Top 50 HIGH-risk customers for the retention team |
| `outputs/churn_scores.db` | SQLite database with historical scoring records |
| `reports/DS3_Report.md` | Full written report |
| `reports/model_metrics.csv` | Model comparison table |
| `reports/business_impact.json` | Revenue-at-risk estimates |
| `models/best_model.pkl` | Serialised production model |
| `models/preprocessor.pkl` | Fitted preprocessing pipeline |

## Model Performance

| Model | AUC-ROC | F1 |
|---|---|---|
| Logistic Regression | **0.8465** | 0.6120 |
| XGBoost | 0.8366 | 0.6305 |
| LightGBM | 0.8334 | 0.6130 |
