"""
Teyzix Core – DS-3
Customer Churn Prediction Dashboard
Run: streamlit run dashboard/app.py
"""

import os, json, sqlite3, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import streamlit as st

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = f"{BASE}/outputs"
REP_DIR = f"{BASE}/reports"
DB_PATH = f"{BASE}/outputs/churn_scores.db"

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="Churn Risk Dashboard – Teyzix Core",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction & Retention Dashboard")
st.caption("Teyzix Core Internship · DS-3 · Weekly Scoring System")
st.markdown("---")

# ── Load data ────────────────────────────────
@st.cache_data
def load_scores():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM churn_scores ORDER BY churn_probability DESC", conn)
    conn.close()
    return df

@st.cache_data
def load_metrics():
    return pd.read_csv(f"{REP_DIR}/model_metrics.csv")

@st.cache_data
def load_impact():
    with open(f"{REP_DIR}/business_impact.json") as f:
        return json.load(f)

df      = load_scores()
metrics = load_metrics()
impact  = load_impact()

# ── KPI cards ────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("🔴 HIGH Risk",   f"{(df['risk_level']=='HIGH').sum():,}")
col3.metric("🟡 MEDIUM Risk", f"{(df['risk_level']=='MEDIUM').sum():,}")
col4.metric("🟢 LOW Risk",    f"{(df['risk_level']=='LOW').sum():,}")
col5.metric("💰 Est. Revenue Saved", f"${impact['Total Estimated Revenue Saved (USD)']:,.0f}")

st.markdown("---")

# ── Row 1: Risk Distribution + Model Metrics ─
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Weekly Risk Segmentation")
    risk_cnt = df["risk_level"].value_counts().reindex(["HIGH","MEDIUM","LOW"])
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#e74c3c","#f39c12","#27ae60"]
    bars = ax.bar(risk_cnt.index, risk_cnt.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, risk_cnt.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+15, str(val),
                ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Customers"); ax.set_xlabel("Risk Level")
    ax.set_title("Risk Distribution – Current Week")
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig); plt.close()

with right:
    st.subheader("Model Performance Comparison")
    disp = metrics.set_index("Model")[["AUC-ROC","Precision","Recall","F1"]]
    st.dataframe(
        disp.style.highlight_max(color="#d5f5e3").format("{:.4f}"),
        use_container_width=True
    )
    st.caption("✅ Best model selected by AUC-ROC")

st.markdown("---")

# ── Row 2: Images ────────────────────────────
st.subheader("Model Diagnostics")
c1, c2 = st.columns(2)
with c1:
    if os.path.exists(f"{REP_DIR}/roc_curves.png"):
        st.image(f"{REP_DIR}/roc_curves.png", caption="ROC Curves", use_column_width=True)
with c2:
    if os.path.exists(f"{REP_DIR}/shap_global_importance.png"):
        st.image(f"{REP_DIR}/shap_global_importance.png", caption="SHAP Feature Importance", use_column_width=True)

st.markdown("---")

# ── Row 3: Confusion Matrices ─────────────────
if os.path.exists(f"{REP_DIR}/confusion_matrices.png"):
    st.subheader("Confusion Matrices")
    st.image(f"{REP_DIR}/confusion_matrices.png", use_column_width=True)

st.markdown("---")

# ── Row 4: Customer Drill-Down ────────────────
st.subheader("🔍 Customer Drill-Down")

col_search, col_filter = st.columns([1, 1])
with col_filter:
    risk_filter = st.selectbox("Filter by Risk Level", ["All", "HIGH", "MEDIUM", "LOW"])

filtered = df if risk_filter == "All" else df[df["risk_level"] == risk_filter]

with col_search:
    customer_id = st.selectbox("Select Customer ID", filtered["customerID"].tolist()[:200])

if customer_id:
    row = df[df["customerID"] == customer_id].iloc[0]
    st.markdown(f"### Customer: `{customer_id}`")

    d1, d2, d3 = st.columns(3)
    risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    d1.metric("Churn Probability", f"{row['churn_probability']*100:.1f}%")
    d2.metric("Risk Level", f"{risk_emoji.get(row['risk_level'],'')} {row['risk_level']}")
    d3.metric("Model", row["model_used"])

    st.markdown("**Top Churn Drivers (SHAP)**")
    for i, col in enumerate(["top_reason_1", "top_reason_2", "top_reason_3"], 1):
        if row[col]:
            st.write(f"  {i}. `{row[col]}`")

    st.info(f"**Recommended Action:** {row['recommended_action']}")

st.markdown("---")

# ── Row 5: High Risk Table ────────────────────
st.subheader("🔴 High-Risk Customer List (Top 25)")
high = df[df["risk_level"] == "HIGH"].head(25)[
    ["customerID","churn_probability","top_reason_1","top_reason_2","recommended_action"]
].reset_index(drop=True)
high["churn_probability"] = (high["churn_probability"] * 100).round(1).astype(str) + "%"
st.dataframe(high, use_container_width=True)

# ── Row 6: Business Impact ────────────────────
st.markdown("---")
st.subheader("💼 Business Impact Analysis")
bi_col = st.columns(3)
bi_col[0].metric("HIGH Risk Revenue Saved", f"${impact['Estimated Revenue Saved – HIGH Risk (USD)']:,.0f}")
bi_col[1].metric("MEDIUM Risk Revenue Saved", f"${impact['Estimated Revenue Saved – MEDIUM Risk (USD)']:,.0f}")
bi_col[2].metric("Total Revenue Saved", f"${impact['Total Estimated Revenue Saved (USD)']:,.0f}")
st.caption("*Assumes 30% intervention success rate and 12-month average revenue at risk per customer.*")

st.markdown("---")
st.caption("Teyzix Core Internship | Task DS-3 | Customer Churn Prediction & Retention Scoring System")
