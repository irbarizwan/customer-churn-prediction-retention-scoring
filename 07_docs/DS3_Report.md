# Customer Churn Prediction & Retention Scoring System
**Teyzix Core Internship | Task DS-3 | Data Science**  
**Submission Date:** May 26, 2026

---

## 1. Executive Summary

This report documents an end-to-end machine learning pipeline built to predict customer churn for a telecom provider, assign weekly risk scores, and surface actionable retention recommendations. The system integrates feature engineering, three trained ML models, SHAP-based explainability, automated batch scoring, risk segmentation, and a business impact estimate.

**Key Results:**
- Best model AUC-ROC: **0.8465** (Logistic Regression with engineered features)
- Customers flagged HIGH risk: **1,686** (23.9%)
- Estimated monthly revenue protected: **~$599,175** (at 30% intervention success)
- All 7,043 customers scored and stored with intervention recommendations

---

## 2. Dataset Overview

**Source:** IBM Telco Customer Churn Dataset  
**Records:** 7,043 customers · 21 raw features  
**Target:** `Churn` (Yes / No) — 26.5% churn rate (class imbalance handled via `class_weight="balanced"` and `scale_pos_weight`)

| Feature Type | Examples |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod |
| Services | PhoneService, InternetService, OnlineSecurity, TechSupport |
| Billing | MonthlyCharges, TotalCharges |

**Data Cleaning:**
- `TotalCharges` had whitespace-encoded blanks for new customers (tenure = 0); converted to numeric and imputed with median.
- No other missing values detected.

---

## 3. Feature Engineering

Sixteen new features were derived from the raw data to capture behavioral signals associated with churn:

| Feature | Logic | Business Rationale |
|---|---|---|
| `tenure_stage` | Binned into new/growing/mature/loyal | Lifecycle risk differs across segments |
| `avg_monthly_spend` | TotalCharges / (tenure + 1) | Normalises spend over time |
| `charge_to_tenure_ratio` | MonthlyCharges / (tenure + 1) | Detects recent billing spikes |
| `billing_spike_flag` | MonthlyCharges > 75th percentile | High-charge customers churn more |
| `service_count` | Count of active services | Low adoption → disengagement |
| `low_service_flag` | service_count ≤ 2 | Minimal investment in ecosystem |
| `contract_risk_score` | Month-to-month=2, One year=1, Two year=0 | Short contracts = higher churn risk |
| `payment_risk_score` | Electronic check=3 → Credit card auto=0 | Friction in payment → churn |
| `digital_risk_combo` | Paperless billing AND electronic check | High-risk payment combination |
| `isolated_senior` | Senior, no partner, no dependents | Vulnerable segment |
| `fiber_optic_flag` | InternetService == Fiber optic | Historically high churn in fibre segment |
| `no_tech_support` | TechSupport == No | Lack of help leads to frustration |
| `no_security` | OnlineSecurity == No | Low service engagement |

---

## 4. Model Training & Comparison

Three models were trained using stratified 80/20 train-test split. Class imbalance was addressed in all three models.

### 4.1 Models

**Logistic Regression (Baseline)**  
- `C=0.5`, `class_weight="balanced"`, `max_iter=1000`
- Provides interpretable coefficients; strong regularised baseline

**XGBoost**  
- `n_estimators=300`, `learning_rate=0.05`, `max_depth=5`
- `scale_pos_weight` set to ratio of negative/positive classes
- Gradient-boosted trees with regularisation

**LightGBM**  
- `n_estimators=300`, `learning_rate=0.05`, `num_leaves=40`
- `class_weight="balanced"`
- Histogram-based boosting; fast and memory-efficient

### 4.2 Results

| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | **0.8465** | 0.5043 | 0.7781 | 0.6120 |
| XGBoost | 0.8366 | 0.5377 | 0.7620 | 0.6305 |
| LightGBM | 0.8334 | 0.5216 | 0.7433 | 0.6130 |

**Selected Model:** Logistic Regression — highest AUC-ROC, strong recall (critical for catching churners), interpretable.

> Note: All three models achieved AUC > 0.83, demonstrating that the feature engineering substantially improved signal over raw features. Recall is prioritised over precision in this context because missing a churner (false negative) is more costly than a false alarm.

---

## 5. SHAP Explainability

SHAP (SHapley Additive exPlanations) values were computed for every prediction using `shap.LinearExplainer` on the Logistic Regression model.

### 5.1 Global Feature Importance

Top churn drivers by mean absolute SHAP value:

1. `contract_risk_score` — Month-to-month contract is the single strongest churn signal
2. `fiber_optic_flag` — Fibre internet customers churn at significantly higher rates
3. `no_tech_support` — Customers without tech support are more likely to leave
4. `tenure` — New customers (low tenure) are most at risk
5. `MonthlyCharges` — Higher bills amplify churn probability
6. `digital_risk_combo` — Electronic check + paperless billing is a high-risk combination
7. `payment_risk_score` — Non-automatic payment methods correlate with churn
8. `no_security` — Absence of Online Security service
9. `charge_to_tenure_ratio` — Disproportionately high recent charges
10. `service_count` — Fewer services = less stickiness

### 5.2 Per-Customer Explanation

For each customer the pipeline outputs their top 3 SHAP-driving features and direction of influence. Example:

```
Customer: 7590-VHVEG
  Churn Probability: 72.4%  |  Risk: HIGH
  Top Reason 1: contract_risk_score  (↑ increases churn)
  Top Reason 2: fiber_optic_flag     (↑ increases churn)
  Top Reason 3: no_tech_support      (↑ increases churn)
  Recommendation: Offer 12-month contract upgrade | Escalate to tech-support for fibre review
```

---

## 6. Weekly Batch Scoring Pipeline

The scoring pipeline is modular and designed to be scheduled weekly (e.g., via Apache Airflow or a cron job):

```
raw CSV → feature engineering → preprocessor transform
       → best model predict_proba → SHAP values
       → risk segmentation → intervention recommendation
       → SQLite / PostgreSQL INSERT → CSV export
```

**Risk Segmentation Logic:**

| Churn Probability | Risk Level | Action |
|---|---|---|
| ≥ 70% | HIGH | Immediate retention call required |
| 40% – 69% | MEDIUM | Email campaign / targeted promotion |
| < 40% | LOW | No action — monitor next cycle |

---

## 7. Intervention Recommendation System

Recommendations are generated by mapping the top positive SHAP features to business actions:

| Driving Feature | Recommended Action |
|---|---|
| contract_risk_score high | Offer 12-month contract upgrade with loyalty discount |
| fiber_optic_flag | Escalate to tech-support for fibre quality review |
| no_tech_support | Assign priority support agent follow-up within 48 hrs |
| billing_spike / MonthlyCharges | Offer payment flexibility plan or billing credit |
| low tenure | Onboarding loyalty bonus or service upgrade |
| no_security / no_backup | Bundle free Online Security add-on for 3 months |
| payment_risk_score | Switch to auto-payment with 5% discount incentive |
| fiber internet | Offer discounted data bundle upgrade |

Each customer receives a maximum of two concrete, prioritised actions.

---

## 8. Risk Distribution – Current Week

| Segment | Count | % of Base |
|---|---|---|
| HIGH | 1,686 | 23.9% |
| MEDIUM | 1,768 | 25.1% |
| LOW | 3,589 | 51.0% |

---

## 9. Business Impact Analysis

| Metric | Value |
|---|---|
| Total Customers Scored | 7,043 |
| Avg Monthly Revenue / Customer | $64.76 |
| HIGH Risk – Estimated Revenue Saved | $393,078 |
| MEDIUM Risk – Estimated Revenue Saved | $206,098 |
| **Total Revenue Protected** | **$599,175** |

*Assumptions: 30% intervention success rate on HIGH risk; 15% on MEDIUM risk; 12-month average revenue horizon.*

---

## 10. Dashboard

A Streamlit dashboard (`dashboard/app.py`) provides:
- KPI cards: total customers, risk counts, revenue saved
- Weekly risk distribution bar chart
- Model performance comparison table
- ROC curves and confusion matrix images
- SHAP global feature importance
- Customer drill-down: search by ID, view probability, SHAP reasons, recommendation
- Full HIGH-risk table for the retention team

**To run:**
```bash
streamlit run dashboard/app.py
```

---

## 11. Repository Structure

```
churn_project/
├── data/
│   └── telco_churn.csv
├── models/
│   ├── best_model.pkl
│   └── preprocessor.pkl
├── outputs/
│   ├── churn_scores.db
│   ├── weekly_scores_YYYY-MM-DD.csv
│   └── sample_call_list.csv
├── reports/
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── shap_global_importance.png
│   ├── risk_distribution.png
│   ├── model_metrics.csv
│   └── business_impact.json
├── dashboard/
│   └── app.py
├── pipeline.py
└── README.md
```

---

## 12. Technical Constraints Checklist

| Requirement | Status |
|---|---|
| ≥ 2 ML models beyond baseline | ✅ XGBoost + LightGBM (+ Logistic Regression baseline) |
| SHAP per prediction | ✅ LinearExplainer for all 7,043 customers |
| Weekly batch scoring reproducible | ✅ Fixed random seed, modular pipeline |
| Risk segmentation business-driven | ✅ Probability thresholds tied to action tiers |
| Evaluation metrics clearly reported | ✅ AUC, Precision, Recall, F1, Confusion Matrix |
| Modular pipeline | ✅ Each step independently callable |
| Non-technical interpretable output | ✅ Natural language recommendations in CSV/DB |

---

## 13. Conclusion

The pipeline successfully transforms raw telecom customer data into a production-ready churn intelligence system. Logistic Regression with rich engineered features achieves the best AUC (0.847), and SHAP values provide transparent per-customer explanations that enable the retention team to act on data-driven recommendations rather than intuition. Weekly batch scoring ensures the system stays current, and the Streamlit dashboard delivers insights to both technical and non-technical stakeholders.

---

*Report generated by pipeline.py | Teyzix Core DS-3 | May 2026*
