"""
Model training and evaluation pipeline for:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

from src.data_loader import get_or_create_processed_data
from src.feature_engineering import FEATURE_COLUMNS, FEATURE_DISPLAY_NAMES, prepare_features


def train_and_evaluate_all(
    orders_df=None,
    models_dir="models",
    test_size=0.2,
    random_state=42
):
    """
    Trains Logistic Regression, Random Forest, and SVM models,
    evaluates them comprehensively, and saves artifacts.
    """
    os.makedirs(models_dir, exist_ok=True)

    if orders_df is None:
        orders_df = get_or_create_processed_data()

    X = prepare_features(orders_df)
    y = orders_df["is_cancelled"]

    print(f"Dataset shape for training: {X.shape}, cancellation rate: {y.mean():.4f}")

    # Train / Test split stratified by cancellation target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standard Scaler for LR and SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    results = {}
    curves_data = {}
    models = {}

    # 1. LOGISTIC REGRESSION
    print("\n[1/3] Training Logistic Regression...")
    t0 = time.time()
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        random_state=random_state
    )
    lr.fit(X_train_scaled, y_train)
    lr_train_time = time.time() - t0

    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    models["logistic_regression"] = lr
    joblib.dump(lr, os.path.join(models_dir, "logistic_regression.joblib"))

    # 2. RANDOM FOREST
    print("\n[2/3] Training Random Forest Classifier...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        min_samples_split=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0

    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    models["random_forest"] = rf
    joblib.dump(rf, os.path.join(models_dir, "random_forest.joblib"))

    # 3. SUPPORT VECTOR MACHINE (SVM)
    # Using Calibrated LinearSVC for fast linear margin optimization with exact calibrated probabilities
    print("\n[3/3] Training Support Vector Machine (SVM)...")
    t0 = time.time()
    base_svm = LinearSVC(
        dual=False,
        class_weight="balanced",
        C=0.8,
        max_iter=3000,
        random_state=random_state
    )
    svm = CalibratedClassifierCV(base_svm, cv=3)
    svm.fit(X_train_scaled, y_train)
    svm_train_time = time.time() - t0

    svm_preds = svm.predict(X_test_scaled)
    svm_probs = svm.predict_proba(X_test_scaled)[:, 1]
    models["svm"] = svm
    joblib.dump(svm, os.path.join(models_dir, "svm.joblib"))

    # Performance Evaluation
    model_outputs = {
        "Logistic Regression": {
            "key": "logistic_regression",
            "preds": lr_preds,
            "probs": lr_probs,
            "time": lr_train_time
        },
        "Random Forest": {
            "key": "random_forest",
            "preds": rf_preds,
            "probs": rf_probs,
            "time": rf_train_time
        },
        "Support Vector Machine (SVM)": {
            "key": "svm",
            "preds": svm_preds,
            "probs": svm_probs,
            "time": svm_train_time
        }
    }

    metrics_summary = {}

    for name, data in model_outputs.items():
        preds = data["preds"]
        probs = data["probs"]

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        auc = float(roc_auc_score(y_test, probs))
        cm = confusion_matrix(y_test, preds).tolist()
        tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)

        # ROC Curve points (downsampled for fast web rendering)
        fpr, tpr, _ = roc_curve(y_test, probs)
        step = max(1, len(fpr) // 100)
        fpr_sampled = fpr[::step].tolist()
        tpr_sampled = tpr[::step].tolist()
        if fpr[-1] not in fpr_sampled:
            fpr_sampled.append(float(fpr[-1]))
            tpr_sampled.append(float(tpr[-1]))

        # Precision-Recall Curve points
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs)
        step_pr = max(1, len(prec_curve) // 100)
        prec_sampled = prec_curve[::step_pr].tolist()
        rec_sampled = rec_curve[::step_pr].tolist()

        metrics_summary[name] = {
            "model_key": data["key"],
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "confusion_breakdown": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp
            },
            "training_time_seconds": round(data["time"], 3),
            "classification_report": report,
            "roc_curve": {"fpr": fpr_sampled, "tpr": tpr_sampled},
            "pr_curve": {"precision": prec_sampled, "recall": rec_sampled}
        }

    # Feature Importance Calculations
    rf_importances = rf.feature_importances_.tolist()
    lr_coefficients = lr.coef_[0].tolist()

    feature_meta = []
    for idx, col in enumerate(FEATURE_COLUMNS):
        feature_meta.append({
            "feature": col,
            "display_name": FEATURE_DISPLAY_NAMES.get(col, col),
            "rf_importance": round(rf_importances[idx], 4),
            "lr_coefficient": round(lr_coefficients[idx], 4)
        })

    # Sort by RF importance
    feature_meta_sorted = sorted(feature_meta, key=lambda x: x["rf_importance"], reverse=True)

    metadata = {
        "dataset_total_orders": len(orders_df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_cancellation_count": int(y_test.sum()),
        "test_fulfilled_count": int((y_test == 0).sum()),
        "feature_columns": FEATURE_COLUMNS,
        "feature_display_names": FEATURE_DISPLAY_NAMES,
        "feature_importances": feature_meta_sorted,
        "metrics": metrics_summary,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save a rich test sample for user batch testing in Streamlit
    sample_df = orders_df.sample(min(200, len(orders_df)), random_state=42).copy()
    sample_file = os.path.join("data", "sample_test_orders.csv")
    sample_df[
        ["InvoiceNo", "n_unique_items", "total_quantity", "total_amount",
         "avg_unit_price", "max_unit_price", "country", "hour", "day_of_week",
         "month", "is_cancelled"]
    ].to_csv(sample_file, index=False)
    print(f"Sample test batch saved to {sample_file}")

    # Generate precomputed EDA stats for lightning fast Streamlit load
    generate_precomputed_eda(orders_df, models_dir)

    print("\nTraining completed successfully! All artifacts saved.")
    return metadata


def generate_precomputed_eda(orders_df, models_dir="models"):
    """
    Computes summary aggregations for EDA tab in Streamlit.
    """
    total_orders = int(len(orders_df))
    cancelled_orders = int(orders_df["is_cancelled"].sum())
    fulfilled_orders = total_orders - cancelled_orders
    cancellation_rate = round(cancelled_orders / total_orders * 100, 2)
    total_revenue = float(orders_df[orders_df["is_cancelled"] == 0]["total_amount"].sum())
    lost_revenue = float(orders_df[orders_df["is_cancelled"] == 1]["total_amount"].sum())
    unique_customers = int(orders_df["customer_id"].dropna().nunique())

    # Hourly distribution
    hourly = orders_df.groupby("hour")["is_cancelled"].agg(
        total="count",
        cancelled="sum"
    ).reset_index()
    hourly["cancellation_rate"] = (hourly["cancelled"] / hourly["total"] * 100).round(2)

    # Day of week distribution
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = orders_df.groupby("day_of_week")["is_cancelled"].agg(
        total="count",
        cancelled="sum"
    ).reset_index()
    dow["day_name"] = dow["day_of_week"].apply(lambda d: day_names[d] if d < len(day_names) else str(d))
    dow["cancellation_rate"] = (dow["cancelled"] / dow["total"] * 100).round(2)

    # Top countries
    country_summary = orders_df.groupby("country").agg(
        total_orders=("InvoiceNo", "count"),
        cancelled_orders=("is_cancelled", "sum"),
        total_spend=("total_amount", "sum")
    ).reset_index()
    country_summary["cancellation_rate"] = (
        country_summary["cancelled_orders"] / country_summary["total_orders"] * 100
    ).round(2)
    top_countries = country_summary.sort_values(by="total_orders", ascending=False).head(10).to_dict(orient="records")

    eda_payload = {
        "kpis": {
            "total_orders": total_orders,
            "fulfilled_orders": fulfilled_orders,
            "cancelled_orders": cancelled_orders,
            "cancellation_rate": cancellation_rate,
            "total_revenue": round(total_revenue, 2),
            "lost_revenue": round(lost_revenue, 2),
            "unique_customers": unique_customers
        },
        "hourly": hourly.to_dict(orient="records"),
        "day_of_week": dow.to_dict(orient="records"),
        "top_countries": top_countries
    }

    with open(os.path.join(models_dir, "precomputed_eda.json"), "w") as f:
        json.dump(eda_payload, f, indent=2)


if __name__ == "__main__":
    train_and_evaluate_all()
