# 🛒 E-Commerce Order Cancellation Prediction (OrderSense AI)

A simple, production-grade Machine Learning platform to detect, evaluate, and predict e-commerce order cancellations before warehouse fulfillment. Powered by **Logistic Regression**, **Random Forest Classifier**, and **Support Vector Machine (SVM)** algorithms, paired with an interactive, professional **Streamlit** user interface.

---

## 📌 Executive Summary

Order cancellations cause substantial operational waste in e-commerce—including pick-and-pack labor, wasted packaging, dispatch transit costs, and reverse logistics fees.

Using 540,000+ transaction lines and 25,900 aggregated order invoices, this project extracts order-level behavioral signals and evaluates cancellation risk with high Recall (>90%) and strong ROC-AUC (>0.97).

---

## 🤖 Algorithms & Model Performance Comparison

Evaluated on a stratified test holdout of **5,180 orders** (baseline cancellation rate = 14.81%):

| Algorithm | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** *(Recommended)* | **94.56%** | **76.80%** | **90.61%** | **0.8313** | **0.9767** | 0.78s |
| **Logistic Regression** *(Balanced L2)* | **88.51%** | 56.76% | **94.13%** | 0.7082 | **0.9501** | 0.12s |
| **Support Vector Machine (SVM)** *(Calibrated)* | **91.10%** | 75.25% | 59.45% | 0.6642 | **0.9504** | 0.52s |

### Key Metric Highlights:
- **Random Forest**: Best overall balance with **94.56% Accuracy**, **90.61% Recall**, and **0.9767 ROC-AUC**. Captures complex non-linear combinations of price and basket size.
- **Logistic Regression**: Maximizes detection coverage with an outstanding **94.13% Recall**, catching nearly all prospective cancellations with rapid execution.
- **Support Vector Machine (SVM)**: Maximum-margin boundary with **75.25% Precision** and **0.9504 ROC-AUC**, calibrated via 3-fold Platt scaling for reliable probability estimations.

---

## 🌟 Key Application Features

### 1. 🔮 Live Order Cancellation Predictor (Default Landing View)
- Realistic one-click preset test scenarios (*Standard Retail Basket*, *High-Price Single Item*, *Chronic Canceller Customer*, *Wholesale Export Order*).
- Interactive inputs for basket size, quantity, total value, item pricing, hour, day, country, and customer history.
- Dynamic cancellation probability meter (0 - 100%) and color-coded risk badge:
  - **Low Risk (<30%)**: Green badge, safe to dispatch immediately.
  - **Moderate Risk (30-65%)**: Amber badge, trigger automated customer verification.
  - **High Risk (>65%)**: Red badge, hold fulfillment for fraud/address check.
- Specific risk factor explanation and multi-model consensus prediction.

### 2. 🤖 Model Benchmarks & Metrics Studio
- Side-by-side metric tables and grouped comparative bar charts.
- Interactive Plotly Confusion Matrices with detailed counts (True Negatives, False Positives, False Negatives, True Positives).
- Receiver Operating Characteristic (ROC) curves with individual AUC scores.
- Precision-Recall (PR) curves.
- Feature Importance rankings (Random Forest Gini) and directional Log-Odds weights (Logistic Regression coefficients).

### 3. 📊 Executive Dashboard & Analytics
- High-level KPIs: Total Orders (25,900), Fulfilled Count (22,064), Cancelled Count (3,836), Revenue at Risk (£1.53M).
- Hourly and Day-of-Week cancellation distribution charts.
- Country-level order volume and cancellation rate visualizer.
- Interactive order data table with status filtering.

---

## 🗂️ Project Directory Structure

```
Ecommerce-Order-Cancellation-Prediction/
├── app.py                              # Streamlit web application
├── requirements.txt                    # Project dependencies
├── README.md                           # Documentation & operational guide
├── data/
│   ├── data.csv                        # Raw retail transactions dataset
│   ├── processed_orders.csv            # Cleaned order-level feature dataset
│   └── sample_test_orders.csv          # Sample orders for batch testing
├── models/
│   ├── logistic_regression.joblib      # Trained Logistic Regression model
│   ├── random_forest.joblib            # Trained Random Forest model
│   ├── svm.joblib                      # Trained Calibrated SVM model
│   ├── scaler.joblib                   # Fitted StandardScaler
│   ├── metadata.json                   # Comprehensive metrics, curves & features
│   └── precomputed_eda.json            # Fast-loading dashboard statistics
└── src/
    ├── __init__.py
    ├── data_loader.py                  # Data loading, cleaning & aggregation
    ├── feature_engineering.py          # Feature definitions & input mapping
    ├── model_trainer.py                # Multi-model training and evaluation
    └── predict.py                      # Unified single-order & batch inference
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-train All Models
```bash
python -m src.model_trainer
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 🔒 Constraints
- No Git commands were run during this setup as per explicit instructions.