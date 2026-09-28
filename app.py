"""
E-Commerce Order Cancellation AI Studio
A modern, simple, and professional Streamlit application for predicting order cancellations
using Logistic Regression, Random Forest, and Support Vector Machine (SVM).
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.predict import (
    load_inference_artifacts,
    predict_single_order,
    get_risk_tier
)
from src.feature_engineering import create_sample_order_dict, FEATURE_DISPLAY_NAMES

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="OrderSense AI | Order Cancellation Studio",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Professional UI Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Banner Header */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 22px;
        color: #f8fafc;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle {
        font-size: 13.5px;
        color: #94a3b8;
        line-height: 1.5;
        margin: 0;
    }

    /* Studio Card Panels */
    .studio-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }
    .studio-panel-title {
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 10px;
    }

    /* Hero Risk Display */
    .risk-hero-card {
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border-width: 2px;
        border-style: solid;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .risk-hero-low {
        background: linear-gradient(180deg, #f0fdf4 0%, #dcfce7 100%);
        border-color: #4ade80;
        color: #14532d;
    }
    .risk-hero-mod {
        background: linear-gradient(180deg, #fffbeb 0%, #fef3c7 100%);
        border-color: #facc15;
        color: #78350f;
    }
    .risk-hero-high {
        background: linear-gradient(180deg, #fef2f2 0%, #fee2e2 100%);
        border-color: #f87171;
        color: #7f1d1d;
    }

    .risk-score-value {
        font-size: 52px;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.03em;
        margin: 10px 0;
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }

    /* Decision Banner */
    .decision-banner {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 18px;
        margin-top: 14px;
        border-left: 5px solid;
        font-size: 13.5px;
        font-weight: 600;
        line-height: 1.5;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }

    /* Consensus Bar Component */
    .consensus-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .consensus-name {
        font-size: 13px;
        font-weight: 600;
        color: #334155;
    }
    .consensus-pct {
        font-size: 14px;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Receipt preview card */
    .receipt-box {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 14px 18px;
        font-size: 12.5px;
        color: #475569;
    }
    .receipt-row {
        display: flex;
        justify-content: space-between;
        padding: 3px 0;
    }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 2px 6px -1px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -2px rgba(0, 0, 0, 0.06);
    }
    .kpi-label {
        font-size: 11.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 11.5px;
        color: #94a3b8;
        margin-top: 5px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 14px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 4px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 700;
        padding: 12px 24px;
        border-radius: 10px 10px 0 0;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 3px solid #2563eb !important;
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .algo-tag {
        background: #f1f5f9;
        color: #334155;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Metadata & EDA Data
# ---------------------------------------------------------
@st.cache_data
def load_metadata():
    path = os.path.join("models", "metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_eda_data():
    path = os.path.join("models", "precomputed_eda.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_sample_orders():
    path = os.path.join("data", "sample_test_orders.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

metadata = load_metadata()
eda_data = load_eda_data()
sample_test_orders = load_sample_orders()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=48)
    st.markdown("### **OrderSense AI**")
    st.caption("E-Commerce Order Cancellation Intelligence")
    st.markdown("---")

    st.markdown("#### **Active ML Algorithms**")
    st.markdown("""
    - <span class='algo-tag'>Logistic Regression</span>
    - <span class='algo-tag'>Random Forest</span>
    - <span class='algo-tag'>Support Vector Machine</span>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### **Quick Guide**")
    st.markdown("""
    1. **Live Predictor**: Select a quick scenario or configure order parameters to assess risk.
    2. **Model Metrics**: View side-by-side accuracy, ROC curves, and confusion matrices.
    3. **Executive Dashboard**: Explore cancellation patterns across hours, days, and countries.
    """)

    st.markdown("---")
    if metadata:
        st.caption(f"**Total Orders:** {metadata.get('dataset_total_orders', 25900):,}")
        st.caption(f"**Test Holdout:** {metadata.get('test_samples', 5180):,} orders")
        st.caption(f"**Pipeline Status:** Active & Cached")


# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-title">
        <span>🛒</span> OrderSense AI: E-Commerce Cancellation Prediction Studio
    </div>
    <div class="hero-subtitle">
        Intelligent order risk scoring engine powered by <b>Logistic Regression</b>, <b>Random Forest</b>, and <b>Support Vector Machine (SVM)</b>.
        Evaluates basket composition, pricing anomalies, and customer history to preempt costly dispatch cancellations.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MAIN HORIZONTAL TABS (PREDICTION FIRST AS REQUESTED)
# =========================================================
tab_pred, tab_models, tab_dash = st.tabs([
    "🔮 Live Order Predictor",
    "🤖 Model Benchmark & Metrics",
    "📊 Executive Dashboard & Analytics"
])


# =========================================================
# TAB 1: LIVE ORDER PREDICTOR (IMPROVED UI)
# =========================================================
with tab_pred:
    # 1. Quick Scenario Presets Showcase
    st.markdown("##### ⚡ Quick Test Scenarios")
    s1, s2, s3, s4 = st.columns(4)

    default_vals = {
        "n_items": 4,
        "quantity": 12,
        "amount": 160.0,
        "avg_price": 13.5,
        "max_price": 28.0,
        "hour": 14,
        "dow": 2,
        "month": 10,
        "is_uk": 1,
        "has_cust_id": 1,
        "cust_orders": 6,
        "cust_cancel_rate": 0.0
    }

    with s1:
        if st.button("🛒 **Standard Retail Basket**\n\n*Multi-item, verified user*", use_container_width=True):
            st.session_state["p_items"] = 5
            st.session_state["p_qty"] = 14
            st.session_state["p_amt"] = 95.0
            st.session_state["p_avg_price"] = 6.8
            st.session_state["p_max_price"] = 15.0
            st.session_state["p_cust_orders"] = 8
            st.session_state["p_cancel_rate"] = 0.0
            st.session_state["p_cust_id"] = 1

    with s2:
        if st.button("🚨 **Luxury Single Item**\n\n*High £295 item, new buyer*", use_container_width=True):
            st.session_state["p_items"] = 1
            st.session_state["p_qty"] = 1
            st.session_state["p_amt"] = 295.0
            st.session_state["p_avg_price"] = 295.0
            st.session_state["p_max_price"] = 295.0
            st.session_state["p_cust_orders"] = 1
            st.session_state["p_cancel_rate"] = 0.0
            st.session_state["p_cust_id"] = 0

    with s3:
        if st.button("⚠️ **Chronic Canceller**\n\n*75% past cancel rate*", use_container_width=True):
            st.session_state["p_items"] = 2
            st.session_state["p_qty"] = 4
            st.session_state["p_amt"] = 120.0
            st.session_state["p_avg_price"] = 30.0
            st.session_state["p_max_price"] = 65.0
            st.session_state["p_cust_orders"] = 4
            st.session_state["p_cancel_rate"] = 0.75
            st.session_state["p_cust_id"] = 1

    with s4:
        if st.button("📦 **Wholesale Bulk Order**\n\n*24 items, £1,450 total*", use_container_width=True):
            st.session_state["p_items"] = 24
            st.session_state["p_qty"] = 380
            st.session_state["p_amt"] = 1450.0
            st.session_state["p_avg_price"] = 3.8
            st.session_state["p_max_price"] = 12.0
            st.session_state["p_cust_orders"] = 15
            st.session_state["p_cancel_rate"] = 0.0
            st.session_state["p_cust_id"] = 1

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Main Two-Column Split Studio
    col_config, col_result = st.columns([1.1, 1.2], gap="large")

    # ---- LEFT COLUMN: Order Configuration Panel ----
    with col_config:
        st.markdown("""
        <div class="studio-panel-title">
            <span>⚙️</span> Order Configuration & Basket Parameters
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("###### 🛍️ Basket & Pricing Details")
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                n_unique_items = st.number_input(
                    "Unique Items (SKUs):",
                    min_value=1,
                    max_value=200,
                    value=int(st.session_state.get("p_items", default_vals["n_items"])),
                    help="Count of distinct product line-items in the order"
                )
                total_amount = st.number_input(
                    "Total Order Value (£):",
                    min_value=0.5,
                    max_value=50000.0,
                    value=float(st.session_state.get("p_amt", default_vals["amount"])),
                    step=10.0,
                    help="Gross total amount of the order in GBP"
                )
            with c_b2:
                total_quantity = st.number_input(
                    "Total Quantity:",
                    min_value=1,
                    max_value=5000,
                    value=int(st.session_state.get("p_qty", default_vals["quantity"])),
                    help="Sum of quantities across all items"
                )
                max_unit_price = st.number_input(
                    "Max Unit Price (£):",
                    min_value=0.1,
                    max_value=5000.0,
                    value=float(st.session_state.get("p_max_price", default_vals["max_price"])),
                    step=5.0,
                    help="Highest single unit price inside the basket"
                )

            avg_unit_price = total_amount / max(1, total_quantity)
            st.caption(f"💡 Calculated Avg Price per Item: **£{avg_unit_price:.2f}** | Single Item Basket: **{'Yes' if n_unique_items==1 else 'No'}**")

            st.markdown("---")
            st.markdown("###### 👤 Customer & Fulfillment Profile")
            c_c1, c_c2 = st.columns(2)
            with c_c1:
                has_cust_id = st.selectbox(
                    "Customer Account:",
                    [1, 0],
                    format_func=lambda x: "Registered Customer" if x == 1 else "Guest Checkout",
                    index=0 if st.session_state.get("p_cust_id", 1) == 1 else 1
                )
                cust_total_orders = st.number_input(
                    "Prior Order Count:",
                    min_value=1,
                    max_value=100,
                    value=int(st.session_state.get("p_cust_orders", default_vals["cust_orders"]))
                )
            with c_c2:
                is_uk = st.selectbox(
                    "Delivery Country:",
                    [1, 0],
                    format_func=lambda x: "United Kingdom (Domestic)" if x == 1 else "International Export"
                )
                cust_cancel_rate_pct = st.slider(
                    "Past Cancel Rate (%):",
                    min_value=0,
                    max_value=100,
                    value=int(st.session_state.get("p_cancel_rate", default_vals["cust_cancel_rate"]) * 100),
                    step=5
                )

            st.markdown("---")
            st.markdown("###### ⏱️ Timing & Active Model")
            c_t1, c_t2 = st.columns(2)
            with c_t1:
                hour = st.slider("Order Hour (24h):", 0, 23, default_vals["hour"])
                day_of_week = st.selectbox(
                    "Day of Week:",
                    list(range(7)),
                    format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d],
                    index=default_vals["dow"]
                )
            with c_t2:
                month = st.selectbox("Month of Year:", list(range(1, 13)), index=default_vals["month"] - 1)
                model_choice = st.selectbox(
                    "Primary Model:",
                    [
                        ("random_forest", "Random Forest (Recommended)"),
                        ("logistic_regression", "Logistic Regression"),
                        ("svm", "Support Vector Machine (SVM)")
                    ],
                    format_func=lambda x: x[1]
                )[0]

        eval_btn = st.button("⚡ Evaluate Cancellation Risk Now", type="primary", use_container_width=True)

    # ---- RIGHT COLUMN: Executive Prediction Intelligence Panel ----
    with col_result:
        st.markdown("""
        <div class="studio-panel-title">
            <span>🎯</span> Executive Risk Assessment & Decision Output
        </div>
        """, unsafe_allow_html=True)

        # Compute prediction immediately (reactive, updates on button or parameters)
        order_payload = create_sample_order_dict(
            n_unique_items=n_unique_items,
            total_quantity=total_quantity,
            total_amount=total_amount,
            avg_unit_price=avg_unit_price,
            max_unit_price=max_unit_price,
            hour=hour,
            day_of_week=day_of_week,
            month=month,
            is_uk=is_uk,
            has_customer_id=has_cust_id,
            cust_total_orders=cust_total_orders,
            cust_cancel_rate=cust_cancel_rate_pct / 100.0
        )

        res = predict_single_order(order_payload, model_key=model_choice)
        prob_pct = res["probability_percent"]
        tier = res["risk_tier"]
        badge_color = res["badge_color"]
        badge_bg = res["badge_bg"]

        # 1. Hero Risk Banner
        hero_class = "risk-hero-low" if "Low" in tier else ("risk-hero-mod" if "Moderate" in tier else "risk-hero-high")
        border_col = "#10b981" if "Low" in tier else ("#f59e0b" if "Moderate" in tier else "#ef4444")

        st.markdown(f"""
        <div class="risk-hero-card {hero_class}">
            <div style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.85;">
                Predicted Cancellation Risk
            </div>
            <div class="risk-score-value">
                {prob_pct}%
            </div>
            <div>
                <span class="risk-badge" style="background-color: {badge_color}; color: #ffffff;">
                    {tier}
                </span>
            </div>
            <div style="font-size: 12px; margin-top: 10px; opacity: 0.85;">
                Evaluated by <b>{model_choice.replace('_', ' ').title()}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Operational Action Recommendation Box
        action_icon = "✅" if "Low" in tier else ("⚠️" if "Moderate" in tier else "🛑")
        st.markdown(f"""
        <div class="decision-banner" style="border-left-color: {border_col};">
            <span style="font-size: 16px; margin-right: 6px;">{action_icon}</span>
            <b>Operational Directive:</b> {res['action']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Cross-Algorithm Consensus Card
        st.markdown("###### 🤝 Cross-Model Consensus Triad")
        comp = res["model_comparison"]
        c_rf_val = comp.get("Random Forest", 0)
        c_lr_val = comp.get("Logistic Regression", 0)
        c_svm_val = comp.get("Support Vector Machine (SVM)", 0)

        st.markdown(f"""
        <div class="consensus-item">
            <span class="consensus-name">🌲 Random Forest (Ensemble Trees)</span>
            <span class="consensus-pct" style="color: {'#10b981' if c_rf_val<30 else ('#f59e0b' if c_rf_val<65 else '#ef4444')};">{c_rf_val}%</span>
        </div>
        <div class="consensus-item">
            <span class="consensus-name">📈 Logistic Regression (Log-Odds)</span>
            <span class="consensus-pct" style="color: {'#10b981' if c_lr_val<30 else ('#f59e0b' if c_lr_val<65 else '#ef4444')};">{c_lr_val}%</span>
        </div>
        <div class="consensus-item">
            <span class="consensus-name">🎯 Support Vector Machine (Margin Boundary)</span>
            <span class="consensus-pct" style="color: {'#10b981' if c_svm_val<30 else ('#f59e0b' if c_svm_val<65 else '#ef4444')};">{c_svm_val}%</span>
        </div>
        """, unsafe_allow_html=True)

        # 4. Key Contributing Risk Factors
        st.markdown("###### 🔍 Risk Factor Diagnostics")
        for factor in res["risk_factors"]:
            st.info(f"• {factor}")

        # 5. Order Mini Receipt
        st.markdown("###### 🧾 Order Snapshot")
        st.markdown(f"""
        <div class="receipt-box">
            <div class="receipt-row"><span>Total Value:</span> <b>£{total_amount:.2f}</b></div>
            <div class="receipt-row"><span>Unique SKUs:</span> <b>{n_unique_items}</b></div>
            <div class="receipt-row"><span>Total Units:</span> <b>{total_quantity}</b></div>
            <div class="receipt-row"><span>Max Item Price:</span> <b>£{max_unit_price:.2f}</b></div>
            <div class="receipt-row"><span>Customer Status:</span> <b>{'Registered Account' if has_cust_id==1 else 'Guest'}</b></div>
            <div class="receipt-row"><span>Destination:</span> <b>{'UK Domestic' if is_uk==1 else 'Export'}</b></div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TAB 2: MODEL BENCHMARK & METRICS
# =========================================================
with tab_models:
    st.markdown("### 🤖 Algorithmic Benchmarks & Comprehensive Evaluation Metrics")
    st.caption("Comparison across Logistic Regression, Random Forest Classifier, and Support Vector Machine (SVM).")

    if metadata and "metrics" in metadata:
        metrics = metadata["metrics"]

        # Comparison Cards
        col_lr, col_rf, col_svm = st.columns(3)

        with col_lr:
            m_lr = metrics.get("Logistic Regression", {})
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #3b82f6;">
                <div class="kpi-label">Logistic Regression</div>
                <div class="kpi-value" style="color: #2563eb;">{m_lr.get('accuracy', 0)*100:.2f}%</div>
                <div class="kpi-subtext">
                    <b>ROC-AUC:</b> {m_lr.get('roc_auc', 0):.4f}<br>
                    <b>Recall:</b> {m_lr.get('recall', 0)*100:.2f}%<br>
                    <b>F1-Score:</b> {m_lr.get('f1_score', 0):.4f}<br>
                    <b>Train Time:</b> {m_lr.get('training_time_seconds', 0)}s
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_rf:
            m_rf = metrics.get("Random Forest", {})
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #10b981;">
                <div class="kpi-label">Random Forest <span class="badge-pill" style="background:#dcfce7;color:#15803d;font-size:10px;">Best Overall</span></div>
                <div class="kpi-value" style="color: #059669;">{m_rf.get('accuracy', 0)*100:.2f}%</div>
                <div class="kpi-subtext">
                    <b>ROC-AUC:</b> {m_rf.get('roc_auc', 0):.4f}<br>
                    <b>Recall:</b> {m_rf.get('recall', 0)*100:.2f}%<br>
                    <b>F1-Score:</b> {m_rf.get('f1_score', 0):.4f}<br>
                    <b>Train Time:</b> {m_rf.get('training_time_seconds', 0)}s
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_svm:
            m_svm = metrics.get("Support Vector Machine (SVM)", {})
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 4px solid #8b5cf6;">
                <div class="kpi-label">Support Vector Machine (SVM)</div>
                <div class="kpi-value" style="color: #7c3aed;">{m_svm.get('accuracy', 0)*100:.2f}%</div>
                <div class="kpi-subtext">
                    <b>ROC-AUC:</b> {m_svm.get('roc_auc', 0):.4f}<br>
                    <b>Recall:</b> {m_svm.get('recall', 0)*100:.2f}%<br>
                    <b>F1-Score:</b> {m_svm.get('f1_score', 0):.4f}<br>
                    <b>Train Time:</b> {m_svm.get('training_time_seconds', 0)}s
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed Metrics Views
        m_tab1, m_tab2, m_tab3, m_tab4 = st.tabs([
            "📊 Metrics Comparison Bar Chart",
            "🔲 Confusion Matrices",
            "📈 ROC & Precision-Recall Curves",
            "🌲 Feature Importances"
        ])

        # TAB 2.1: Bar Chart Comparison
        with m_tab1:
            st.markdown("#### 📊 Comparative Metric Matrix")
            records = []
            for m_name, m_val in metrics.items():
                records.append({
                    "Model": m_name,
                    "Accuracy": m_val["accuracy"],
                    "Precision": m_val["precision"],
                    "Recall": m_val["recall"],
                    "F1 Score": m_val["f1_score"],
                    "ROC-AUC": m_val["roc_auc"]
                })
            comp_df = pd.DataFrame(records)

            st.dataframe(
                comp_df.set_index("Model").style.format("{:.4f}").highlight_max(axis=0, color="#dcfce7"),
                use_container_width=True
            )

            melted = comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
            fig_metrics = px.bar(
                melted,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                text="Score",
                color_discrete_map={
                    "Logistic Regression": "#3b82f6",
                    "Random Forest": "#10b981",
                    "Support Vector Machine (SVM)": "#8b5cf6"
                },
                template="plotly_white"
            )
            fig_metrics.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_metrics.update_layout(height=420, yaxis_range=[0.4, 1.05], margin=dict(t=20, b=20))
            st.plotly_chart(fig_metrics, use_container_width=True)

        # TAB 2.2: Confusion Matrices
        with m_tab2:
            st.markdown("#### 🔲 Confusion Matrix Breakdown on Test Orders (n = 5,180)")
            c1, c2, c3 = st.columns(3)

            for idx, (m_name, m_val) in enumerate(metrics.items()):
                target_col = [c1, c2, c3][idx]
                with target_col:
                    cm = np.array(m_val["confusion_matrix"])
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                        x=["Fulfilled (0)", "Cancelled (1)"],
                        y=["Fulfilled (0)", "Cancelled (1)"],
                        color_continuous_scale="Blues",
                        title=f"<b>{m_name}</b>"
                    )
                    fig_cm.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_cm, use_container_width=True)

                    bd = m_val["confusion_breakdown"]
                    st.markdown(f"""
                    - **True Negatives:** {bd['true_negative']} (Correctly passed)
                    - **False Positives:** {bd['false_positive']} (False alarm)
                    - **False Negatives:** {bd['false_negative']} (Missed cancel)
                    - **True Positives:** {bd['true_positive']} (Caught cancel)
                    """)

        # TAB 2.3: ROC & PR Curves
        with m_tab3:
            col_roc, col_pr = st.columns(2)

            with col_roc:
                st.markdown("#### 📈 Receiver Operating Characteristic (ROC) Curves")
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    line=dict(dash="dash", color="#94a3b8"),
                    name="Random Chance (AUC = 0.50)"
                ))

                colors = {
                    "Logistic Regression": "#3b82f6",
                    "Random Forest": "#10b981",
                    "Support Vector Machine (SVM)": "#8b5cf6"
                }

                for m_name, m_val in metrics.items():
                    roc_data = m_val.get("roc_curve", {})
                    if roc_data:
                        fig_roc.add_trace(go.Scatter(
                            x=roc_data.get("fpr", []),
                            y=roc_data.get("tpr", []),
                            mode="lines",
                            line=dict(color=colors.get(m_name, "#475569"), width=2.5),
                            name=f"{m_name} (AUC = {m_val['roc_auc']:.4f})"
                        ))

                fig_roc.update_layout(
                    xaxis_title="False Positive Rate (1 - Specificity)",
                    yaxis_title="True Positive Rate (Recall / Sensitivity)",
                    template="plotly_white",
                    height=420,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            with col_pr:
                st.markdown("#### 📉 Precision-Recall (PR) Curves")
                fig_pr = go.Figure()

                for m_name, m_val in metrics.items():
                    pr_data = m_val.get("pr_curve", {})
                    if pr_data:
                        fig_pr.add_trace(go.Scatter(
                            x=pr_data.get("recall", []),
                            y=pr_data.get("precision", []),
                            mode="lines",
                            line=dict(color=colors.get(m_name, "#475569"), width=2.5),
                            name=f"{m_name}"
                        ))

                fig_pr.update_layout(
                    xaxis_title="Recall (Sensitivity)",
                    yaxis_title="Precision (Positive Predictive Value)",
                    template="plotly_white",
                    height=420,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_pr, use_container_width=True)

        # TAB 2.4: Feature Importance
        with m_tab4:
            st.markdown("#### 🌲 Feature Importance & Coefficient Impact")
            feat_list = metadata.get("feature_importances", [])
            if feat_list:
                df_feat = pd.DataFrame(feat_list)
                c_fi, c_coef = st.columns(2)

                with c_fi:
                    st.markdown("##### 🌲 Random Forest Feature Importance (Gini)")
                    fig_fi = px.bar(
                        df_feat.head(10).sort_values("rf_importance"),
                        x="rf_importance",
                        y="display_name",
                        orientation="h",
                        labels={"rf_importance": "Relative Importance", "display_name": "Feature"},
                        color="rf_importance",
                        color_continuous_scale="Greens",
                        template="plotly_white"
                    )
                    fig_fi.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
                    st.plotly_chart(fig_fi, use_container_width=True)

                with c_coef:
                    st.markdown("##### ⚖️ Logistic Regression Directional Weights")
                    fig_coef = px.bar(
                        df_feat.head(10).sort_values("lr_coefficient"),
                        x="lr_coefficient",
                        y="display_name",
                        orientation="h",
                        labels={"lr_coefficient": "Log-Odds Weight", "display_name": "Feature"},
                        color="lr_coefficient",
                        color_continuous_scale="Spectral",
                        template="plotly_white"
                    )
                    fig_coef.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
                    st.plotly_chart(fig_coef, use_container_width=True)


# =========================================================
# TAB 3: EXECUTIVE DASHBOARD & ANALYTICS
# =========================================================
with tab_dash:
    st.markdown("### 📊 Executive Summary & Key Performance Indicators")
    st.caption("Aggregated analytics from 25,900 e-commerce orders and 540,000+ line-item transactions.")

    if eda_data:
        kpis = eda_data.get("kpis", {})
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Analyzed Orders</div>
                <div class="kpi-value">{kpis.get('total_orders', 25900):,}</div>
                <div class="kpi-subtext">Unique customer invoices</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Fulfilled Orders</div>
                <div class="kpi-value" style="color: #10b981;">{kpis.get('fulfilled_orders', 22064):,}</div>
                <div class="kpi-subtext">{(100 - kpis.get('cancellation_rate', 14.81)):.1f}% completion rate</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Cancelled Orders</div>
                <div class="kpi-value" style="color: #ef4444;">{kpis.get('cancelled_orders', 3836):,}</div>
                <div class="kpi-subtext">Baseline rate: <b>{kpis.get('cancellation_rate', 14.81)}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Gross Value at Risk</div>
                <div class="kpi-value" style="color: #f59e0b;">£{kpis.get('lost_revenue', 1530000):,.0f}</div>
                <div class="kpi-subtext">Cumulative cancelled amount</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Section
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### 🕒 Cancellation Rate by Hour of Day")
            df_hourly = pd.DataFrame(eda_data.get("hourly", []))
            if not df_hourly.empty:
                fig_hour = px.bar(
                    df_hourly,
                    x="hour",
                    y="cancellation_rate",
                    text="cancellation_rate",
                    labels={"hour": "Order Hour (24h)", "cancellation_rate": "Cancellation Rate (%)"},
                    color="cancellation_rate",
                    color_continuous_scale="Reds",
                    template="plotly_white"
                )
                fig_hour.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_hour.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
                st.plotly_chart(fig_hour, use_container_width=True)

        with col_right:
            st.markdown("#### 📅 Cancellation Rate by Day of Week")
            df_dow = pd.DataFrame(eda_data.get("day_of_week", []))
            if not df_dow.empty:
                fig_dow = px.bar(
                    df_dow,
                    x="day_name",
                    y="cancellation_rate",
                    text="cancellation_rate",
                    labels={"day_name": "Day of Week", "cancellation_rate": "Cancellation Rate (%)"},
                    color="cancellation_rate",
                    color_continuous_scale="Blues",
                    template="plotly_white"
                )
                fig_dow.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_dow.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
                st.plotly_chart(fig_dow, use_container_width=True)

        # Geographic Breakdown
        st.markdown("#### 🌍 Top 10 Purchasing Countries & Cancellation Rates")
        top_c = pd.DataFrame(eda_data.get("top_countries", []))
        if not top_c.empty:
            fig_country = px.bar(
                top_c,
                x="country",
                y="cancellation_rate",
                hover_data=["total_orders", "cancelled_orders", "total_spend"],
                text="cancellation_rate",
                labels={"country": "Country", "cancellation_rate": "Cancellation Rate (%)"},
                color="total_orders",
                color_continuous_scale="Purples",
                template="plotly_white"
            )
            fig_country.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_country.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_country, use_container_width=True)

        # Dataset Sample Explorer
        st.markdown("#### 🔎 Interactive Order Records Explorer")
        if not sample_test_orders.empty:
            status_filter = st.radio(
                "Filter records by status:",
                ["All Orders", "Cancelled Only (Target = 1)", "Fulfilled Only (Target = 0)"],
                horizontal=True
            )
            display_df = sample_test_orders.copy()
            if status_filter == "Cancelled Only (Target = 1)":
                display_df = display_df[display_df["is_cancelled"] == 1]
            elif status_filter == "Fulfilled Only (Target = 0)":
                display_df = display_df[display_df["is_cancelled"] == 0]

            st.dataframe(
                display_df.head(50),
                use_container_width=True,
                height=260
            )


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 12px;">
    OrderSense AI Studio • Built for E-Commerce Order Cancellation Intelligence • Powered by Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)
