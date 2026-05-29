"""
Teyzix Core Internship – DS-3
Customer Churn Prediction & Retention Scoring System
Author: [Your Name]
Date: May 2026
"""

import warnings
warnings.filterwarnings("ignore")

import os, json, sqlite3, pickle, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import xgboost as xgb
import lightgbm as lgb

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE = "/home/claude/churn_project"
DATA_PATH  = f"{BASE}/data/telco_churn.csv"
MODEL_DIR  = f"{BASE}/models"
OUT_DIR    = f"{BASE}/outputs"
REPORT_DIR = f"{BASE}/reports"
DB_PATH    = f"{BASE}/outputs/churn_scores.db"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – Loading & Cleaning Data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

# Fix TotalCharges: some blanks show as spaces
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Binary target
df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['Churn_Binary'].mean()*100:.1f}%")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 – Feature Engineering")
print("=" * 60)

def engineer_features(df):
    d = df.copy()

    # --- Tenure lifecycle stage ---
    d["tenure_stage"] = pd.cut(
        d["tenure"],
        bins=[0, 6, 24, 48, 72],
        labels=["new", "growing", "mature", "loyal"],
        include_lowest=True
    ).astype(str)

    # --- Revenue features ---
    d["avg_monthly_spend"]   = d["TotalCharges"] / (d["tenure"] + 1)
    d["charge_to_tenure_ratio"] = d["MonthlyCharges"] / (d["tenure"] + 1)
    d["billing_spike_flag"]  = (
        d["MonthlyCharges"] > d["MonthlyCharges"].quantile(0.75)
    ).astype(int)

    # --- Service bundle depth ---
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    d["service_count"] = 0
    for col in service_cols:
        d["service_count"] += (d[col].isin(["Yes", "DSL", "Fiber optic"])).astype(int)

    d["low_service_flag"] = (d["service_count"] <= 2).astype(int)

    # --- Contract risk ---
    contract_risk = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    d["contract_risk_score"] = d["Contract"].map(contract_risk)

    # --- Payment method risk ---
    pay_risk = {
        "Electronic check": 3,
        "Mailed check": 2,
        "Bank transfer (automatic)": 1,
        "Credit card (automatic)": 0
    }
    d["payment_risk_score"] = d["PaymentMethod"].map(pay_risk).fillna(1)

    # --- Paperless + electronic check flag (high churn combo) ---
    d["digital_risk_combo"] = (
        (d["PaperlessBilling"] == "Yes") &
        (d["PaymentMethod"] == "Electronic check")
    ).astype(int)

    # --- Senior citizen + no partner/dependents (isolation flag) ---
    d["isolated_senior"] = (
        (d["SeniorCitizen"] == 1) &
        (d["Partner"] == "No") &
        (d["Dependents"] == "No")
    ).astype(int)

    # --- Fiber optic flag (high churn segment) ---
    d["fiber_optic_flag"] = (d["InternetService"] == "Fiber optic").astype(int)

    # --- No tech support flag ---
    d["no_tech_support"] = (d["TechSupport"] == "No").astype(int)

    # --- No security flag ---
    d["no_security"] = (d["OnlineSecurity"] == "No").astype(int)

    print(f"  → Engineered features added. Final column count: {d.shape[1]}")
    return d

df = engineer_features(df)

# ─────────────────────────────────────────────
# 3. FEATURE SELECTION & PREPROCESSING
# ─────────────────────────────────────────────
cat_features = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_stage"
]

num_features = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_spend", "charge_to_tenure_ratio", "billing_spike_flag",
    "service_count", "low_service_flag", "contract_risk_score",
    "payment_risk_score", "digital_risk_combo", "isolated_senior",
    "fiber_optic_flag", "no_tech_support", "no_security"
]

TARGET = "Churn_Binary"
ID_COL = "customerID"

X = df[cat_features + num_features]
y = df[TARGET]
ids = df[ID_COL]

# One-hot encode categoricals, scale numerics (with imputers)
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer([
    ("cat", cat_pipeline, cat_features),
    ("num", num_pipeline, num_features)
])

X_enc = preprocessor.fit_transform(X)

# Feature names after encoding
ohe_names = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_features).tolist()
all_feature_names = ohe_names + num_features
print(f"\n  Total features after encoding: {len(all_feature_names)}")

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X_enc, y, ids, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 – Model Training & Comparison")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        C=0.5, class_weight="balanced", max_iter=1000, random_state=42
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder=False, eval_metric="logloss", random_state=42,
        verbosity=0
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=40, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, verbose=-1
    )
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n  Training: {name}")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc   = roc_auc_score(y_test, y_prob)
    prec  = precision_score(y_test, y_pred)
    rec   = recall_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    cm    = confusion_matrix(y_test, y_pred)

    results[name] = {
        "AUC-ROC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "Confusion Matrix": cm.tolist()
    }
    trained_models[name] = model

    print(f"    AUC-ROC={auc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

# ─── Select best model ───
best_name = max(results, key=lambda k: results[k]["AUC-ROC"])
best_model = trained_models[best_name]
print(f"\n  ★ Best model: {best_name} (AUC = {results[best_name]['AUC-ROC']})")

with open(f"{MODEL_DIR}/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open(f"{MODEL_DIR}/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

# ─── Plot ROC curves ───
plt.figure(figsize=(8, 6))
for name, model in trained_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['AUC-ROC']})", lw=2)
plt.plot([0,1],[0,1],"k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/roc_curves.png", dpi=150)
plt.close()
print("  → ROC curve saved.")

# ─── Confusion matrices ───
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(np.array(res["Confusion Matrix"]), annot=True, fmt="d",
                cmap="Blues", ax=ax,
                xticklabels=["No Churn","Churn"], yticklabels=["No Churn","Churn"])
    ax.set_title(f"{name}\n(F1={res['F1']})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/confusion_matrices.png", dpi=150)
plt.close()
print("  → Confusion matrices saved.")

# ─────────────────────────────────────────────
# 5. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 – SHAP Explainability")
print("=" * 60)

# Use tree explainer on best model (XGBoost or LightGBM); fallback to linear
if best_name in ["XGBoost", "LightGBM"]:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    # LightGBM returns list for binary; XGBoost returns 2D array
    if isinstance(shap_values, list):
        sv = shap_values[1]   # class 1 = churn
    else:
        sv = shap_values
else:
    explainer = shap.LinearExplainer(best_model, X_train)
    shap_values = explainer.shap_values(X_test)
    sv = shap_values

print(f"  SHAP values computed. Shape: {sv.shape}")

# Global feature importance
shap_df = pd.DataFrame(np.abs(sv), columns=all_feature_names)
feat_importance = shap_df.mean().sort_values(ascending=False).head(15)

plt.figure(figsize=(9, 6))
feat_importance.sort_values().plot(kind="barh", color="#2e86de")
plt.title(f"Top 15 Global SHAP Feature Importances ({best_name})")
plt.xlabel("Mean |SHAP Value|")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/shap_global_importance.png", dpi=150)
plt.close()
print("  → SHAP global importance plot saved.")

# Per-customer SHAP explanation function
def explain_customer(shap_row, feature_names, top_n=3):
    pairs = sorted(zip(shap_row, feature_names), key=lambda x: abs(x[0]), reverse=True)
    reasons = []
    for val, feat in pairs[:top_n]:
        direction = "increases" if val > 0 else "decreases"
        reasons.append({"feature": feat, "shap_value": round(float(val), 4), "direction": direction})
    return reasons

# ─────────────────────────────────────────────
# 6. WEEKLY BATCH SCORING + RISK SEGMENTATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 – Weekly Batch Scoring & Risk Segmentation")
print("=" * 60)

# Score ALL customers (simulating a weekly batch run)
X_all_enc = preprocessor.transform(X)
prob_all   = best_model.predict_proba(X_all_enc)[:, 1]

# SHAP for all customers
if best_name in ["XGBoost", "LightGBM"]:
    shap_all = explainer.shap_values(X_all_enc)
    if isinstance(shap_all, list):
        sv_all = shap_all[1]
    else:
        sv_all = shap_all
else:
    sv_all = explainer.shap_values(X_all_enc)

# Risk Segmentation
def assign_risk(p):
    if p >= 0.70:
        return "HIGH"
    elif p >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"

# Build scored output
def generate_recommendation(reasons, risk_level):
    """Rule-based recommendations driven by SHAP top reasons."""
    top_features = [r["feature"].lower() for r in reasons if r["direction"] == "increases"]

    if risk_level == "LOW":
        return "No action required – monitor monthly."

    actions = []
    if any("contract" in f for f in top_features):
        actions.append("Offer 12-month contract upgrade with loyalty discount")
    if any("fiber" in f for f in top_features):
        actions.append("Escalate to tech-support for fibre quality review")
    if any("tech_support" in f or "techsupport" in f for f in top_features):
        actions.append("Assign priority support agent follow-up within 48 hrs")
    if any("charge" in f or "billing" in f or "monthly" in f for f in top_features):
        actions.append("Offer payment flexibility plan or billing credit")
    if any("tenure" in f for f in top_features):
        actions.append("Offer onboarding loyalty bonus or service upgrade")
    if any("security" in f or "backup" in f for f in top_features):
        actions.append("Bundle free Online Security add-on for 3 months")
    if any("payment" in f for f in top_features):
        actions.append("Switch to auto-payment with 5% discount incentive")
    if any("internet" in f for f in top_features):
        actions.append("Offer discounted data bundle upgrade")

    if not actions:
        if risk_level == "HIGH":
            actions.append("Immediate retention call – offer personalised discount")
        else:
            actions.append("Send targeted email campaign with service bundle offer")

    return " | ".join(actions[:2])   # max 2 actions

# Re-run with function defined
scored = []
for i, (cid, prob) in enumerate(zip(ids, prob_all)):
    risk    = assign_risk(prob)
    reasons = explain_customer(sv_all[i], all_feature_names, top_n=3)
    top3    = [r["feature"] for r in reasons]
    action  = generate_recommendation(reasons, risk)
    scored.append({
        "customerID": cid,
        "churn_probability": round(float(prob), 4),
        "risk_level": risk,
        "top_reason_1": top3[0] if len(top3) > 0 else "",
        "top_reason_2": top3[1] if len(top3) > 1 else "",
        "top_reason_3": top3[2] if len(top3) > 2 else "",
        "recommended_action": action,
        "score_date": datetime.date.today().isoformat(),
        "model_used": best_name
    })

scored_df = pd.DataFrame(scored)
print(f"  Scored {len(scored_df)} customers")
print(scored_df["risk_level"].value_counts())

# Save to CSV
scored_df.to_csv(f"{OUT_DIR}/weekly_scores_{datetime.date.today()}.csv", index=False)
print(f"  → Weekly scores saved to CSV.")

# ─────────────────────────────────────────────
# 7. SQLITE DATABASE (PostgreSQL alternative)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 – Storing Scores in Database")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS churn_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customerID TEXT,
    churn_probability REAL,
    risk_level TEXT,
    top_reason_1 TEXT,
    top_reason_2 TEXT,
    top_reason_3 TEXT,
    recommended_action TEXT,
    score_date TEXT,
    model_used TEXT
)
""")

cursor.execute("DELETE FROM churn_scores WHERE score_date = ?", (datetime.date.today().isoformat(),))
scored_df.to_sql("churn_scores", conn, if_exists="append", index=False)
conn.commit()

# Verify
count = cursor.execute("SELECT COUNT(*) FROM churn_scores").fetchone()[0]
print(f"  → {count} records stored in SQLite database.")
conn.close()

# ─────────────────────────────────────────────
# 8. SAMPLE CALL LIST (HIGH RISK customers)
# ─────────────────────────────────────────────
high_risk = scored_df[scored_df["risk_level"] == "HIGH"].sort_values(
    "churn_probability", ascending=False
).head(50)

# Anonymize for sample call list
call_list = high_risk.copy()
call_list["customerID"] = call_list["customerID"].apply(lambda x: "CUST-" + x[-4:])
call_list.to_csv(f"{OUT_DIR}/sample_call_list.csv", index=False)
print(f"  → Sample call list: {len(call_list)} high-risk customers saved.")

# ─────────────────────────────────────────────
# 9. MODEL PERFORMANCE TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 – Model Comparison Report")
print("=" * 60)

metrics_df = pd.DataFrame({
    k: {m: v for m, v in v.items() if m != "Confusion Matrix"}
    for k, v in results.items()
}).T.reset_index().rename(columns={"index": "Model"})
metrics_df.to_csv(f"{REPORT_DIR}/model_metrics.csv", index=False)
print(metrics_df.to_string(index=False))

# ─────────────────────────────────────────────
# 10. RISK DISTRIBUTION PLOT
# ─────────────────────────────────────────────
risk_counts = scored_df["risk_level"].value_counts().reindex(["HIGH","MEDIUM","LOW"])
colors = ["#e74c3c","#f39c12","#27ae60"]
plt.figure(figsize=(7, 5))
bars = plt.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, risk_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(val),
             ha="center", va="bottom", fontweight="bold", fontsize=12)
plt.title("Weekly Churn Risk Distribution", fontsize=14, fontweight="bold")
plt.ylabel("Number of Customers")
plt.xlabel("Risk Segment")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}/risk_distribution.png", dpi=150)
plt.close()
print("  → Risk distribution plot saved.")

# ─────────────────────────────────────────────
# 11. BUSINESS IMPACT ANALYSIS
# ─────────────────────────────────────────────
avg_monthly_revenue   = df["MonthlyCharges"].mean()
avg_tenure_remaining  = 12   # assumed months at risk
intervention_success  = 0.30  # 30% retention from intervention
high_risk_count       = (scored_df["risk_level"] == "HIGH").sum()
medium_risk_count     = (scored_df["risk_level"] == "MEDIUM").sum()

revenue_saved_high   = high_risk_count   * avg_monthly_revenue * avg_tenure_remaining * intervention_success
revenue_saved_medium = medium_risk_count * avg_monthly_revenue * avg_tenure_remaining * intervention_success * 0.5

total_revenue_saved = revenue_saved_high + revenue_saved_medium

impact = {
    "Total Customers Scored": len(scored_df),
    "HIGH Risk Customers": int(high_risk_count),
    "MEDIUM Risk Customers": int(medium_risk_count),
    "LOW Risk Customers": int((scored_df["risk_level"] == "LOW").sum()),
    "Avg Monthly Revenue per Customer (USD)": round(avg_monthly_revenue, 2),
    "Assumed Intervention Success Rate": "30%",
    "Estimated Revenue Saved – HIGH Risk (USD)": round(revenue_saved_high, 2),
    "Estimated Revenue Saved – MEDIUM Risk (USD)": round(revenue_saved_medium, 2),
    "Total Estimated Revenue Saved (USD)": round(total_revenue_saved, 2)
}

with open(f"{REPORT_DIR}/business_impact.json", "w") as f:
    json.dump(impact, f, indent=2)

print("\n" + "=" * 60)
print("BUSINESS IMPACT ANALYSIS")
print("=" * 60)
for k, v in impact.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("✓ Pipeline complete. All outputs saved.")
print("=" * 60)
print(f"  → Reports:  {REPORT_DIR}/")
print(f"  → Outputs:  {OUT_DIR}/")
print(f"  → Models:   {MODEL_DIR}/")
