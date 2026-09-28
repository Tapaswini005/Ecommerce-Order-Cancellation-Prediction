"""
Unified prediction and scoring engine for e-commerce order cancellation risk.
Supports single-order what-if analysis and batch CSV inference across
Logistic Regression, Random Forest, and SVM models.
"""

import os
import joblib
import pandas as pd
import numpy as np
from src.feature_engineering import FEATURE_COLUMNS, prepare_features, create_sample_order_dict

_CACHE = {}


def load_inference_artifacts(models_dir="models"):
    """
    Loads models and scalers into memory (cached).
    """
    if "loaded" in _CACHE and _CACHE["models_dir"] == models_dir:
        return _CACHE

    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    lr = joblib.load(os.path.join(models_dir, "logistic_regression.joblib"))
    rf = joblib.load(os.path.join(models_dir, "random_forest.joblib"))
    svm = joblib.load(os.path.join(models_dir, "svm.joblib"))

    _CACHE["scaler"] = scaler
    _CACHE["lr"] = lr
    _CACHE["rf"] = rf
    _CACHE["svm"] = svm
    _CACHE["models_dir"] = models_dir
    _CACHE["loaded"] = True
    return _CACHE


def get_risk_tier(probability):
    """
    Categorizes cancellation probability into actionable business risk tiers.
    """
    if probability < 0.30:
        return {
            "tier": "Low Risk",
            "badge_color": "#10b981",  # Emerald Green
            "badge_bg": "#ecfdf5",
            "action": "Safe to fulfill immediately with standard dispatch.",
            "urgency": "Normal"
        }
    elif probability < 0.65:
        return {
            "tier": "Moderate Risk",
            "badge_color": "#f59e0b",  # Amber Yellow
            "badge_bg": "#fffbeb",
            "action": "Send automated SMS/email confirmation to verify customer intent.",
            "urgency": "Attention"
        }
    else:
        return {
            "tier": "High Risk",
            "badge_color": "#ef4444",  # Crimson Red
            "badge_bg": "#fef2f2",
            "action": "Hold dispatch. Flag for merchant fraud/address verification or instant outreach.",
            "urgency": "Urgent"
        }


def predict_single_order(order_dict, model_key="random_forest", models_dir="models"):
    """
    Evaluates cancellation probability for an individual order.
    """
    artifacts = load_inference_artifacts(models_dir)
    scaler = artifacts["scaler"]
    models = {
        "logistic_regression": artifacts["lr"],
        "random_forest": artifacts["rf"],
        "svm": artifacts["svm"]
    }

    df_input = pd.DataFrame([order_dict])
    X = prepare_features(df_input)
    X_scaled = scaler.transform(X)

    # Get predictions for requested model
    selected_model = models.get(model_key, artifacts["rf"])
    if model_key in ["logistic_regression", "svm"]:
        prob = float(selected_model.predict_proba(X_scaled)[0, 1])
        pred = int(selected_model.predict(X_scaled)[0])
    else:
        prob = float(selected_model.predict_proba(X)[0, 1])
        pred = int(selected_model.predict(X)[0])

    # Also compute comparison across all 3 algorithms
    prob_lr = float(artifacts["lr"].predict_proba(X_scaled)[0, 1])
    prob_rf = float(artifacts["rf"].predict_proba(X)[0, 1])
    prob_svm = float(artifacts["svm"].predict_proba(X_scaled)[0, 1])

    risk = get_risk_tier(prob)

    # Risk factor explanations
    risk_factors = []
    if order_dict.get("n_unique_items", 0) <= 1:
        risk_factors.append("Single-item order: Single item purchases have higher historical return rates.")
    if order_dict.get("max_unit_price", 0) > 50:
        risk_factors.append(f"High item price (£{order_dict.get('max_unit_price'):.2f}): Premium items face higher buyer remorse.")
    if order_dict.get("cust_cancel_rate", 0) > 0.3:
        risk_factors.append(f"Customer history: Customer has a prior {order_dict.get('cust_cancel_rate')*100:.0f}% cancellation rate.")
    if order_dict.get("total_amount", 0) > 500:
        risk_factors.append("High monetary value: Large transactions carry higher scrutiny.")
    if not risk_factors:
        risk_factors.append("Standard order patterns: Healthy basket diversity and regular pricing.")

    return {
        "predicted_class": pred,
        "is_cancelled": bool(pred == 1),
        "probability": round(prob, 4),
        "probability_percent": round(prob * 100, 1),
        "risk_tier": risk["tier"],
        "badge_color": risk["badge_color"],
        "badge_bg": risk["badge_bg"],
        "action": risk["action"],
        "urgency": risk["urgency"],
        "risk_factors": risk_factors,
        "model_comparison": {
            "Logistic Regression": round(prob_lr * 100, 1),
            "Random Forest": round(prob_rf * 100, 1),
            "Support Vector Machine (SVM)": round(prob_svm * 100, 1)
        }
    }


def predict_batch(df_orders, model_key="random_forest", models_dir="models"):
    """
    Runs batch inference on an uploaded DataFrame.
    """
    artifacts = load_inference_artifacts(models_dir)
    scaler = artifacts["scaler"]
    models = {
        "logistic_regression": artifacts["lr"],
        "random_forest": artifacts["rf"],
        "svm": artifacts["svm"]
    }

    X = prepare_features(df_orders)
    X_scaled = scaler.transform(X)

    selected_model = models.get(model_key, artifacts["rf"])
    if model_key in ["logistic_regression", "svm"]:
        probs = selected_model.predict_proba(X_scaled)[:, 1]
        preds = selected_model.predict(X_scaled)
    else:
        probs = selected_model.predict_proba(X)[:, 1]
        preds = selected_model.predict(X)

    result_df = df_orders.copy()
    result_df["predicted_cancellation"] = preds
    result_df["cancellation_probability_%"] = (probs * 100).round(1)
    result_df["risk_tier"] = [get_risk_tier(p)["tier"] for p in probs]
    result_df["recommended_action"] = [get_risk_tier(p)["action"] for p in probs]

    return result_df
