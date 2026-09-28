"""
Feature engineering definitions and transformation utilities.
Ensures consistency between training and live inference.
"""

import pandas as pd
import numpy as np

# Canonical feature list used across all 3 models
FEATURE_COLUMNS = [
    "n_unique_items",
    "total_quantity",
    "avg_item_quantity",
    "total_amount",
    "avg_item_amount",
    "avg_unit_price",
    "max_unit_price",
    "hour",
    "day_of_week",
    "month",
    "is_uk",
    "has_customer_id",
    "items_per_order_ratio",
    "is_high_unit_price",
    "is_single_item",
    "cust_total_orders",
    "cust_cancel_rate"
]

FEATURE_DISPLAY_NAMES = {
    "n_unique_items": "Unique Items (SKU Count)",
    "total_quantity": "Total Order Quantity",
    "avg_item_quantity": "Avg Quantity per Item",
    "total_amount": "Total Order Value (£)",
    "avg_item_amount": "Avg Item Spend (£)",
    "avg_unit_price": "Avg Unit Price (£)",
    "max_unit_price": "Max Unit Price (£)",
    "hour": "Hour of Day (0-23)",
    "day_of_week": "Day of Week (0-6)",
    "month": "Month of Year (1-12)",
    "is_uk": "UK Domestic Order (1=Yes, 0=No)",
    "has_customer_id": "Registered Customer (1=Yes, 0=No)",
    "items_per_order_ratio": "Quantity-to-SKU Ratio",
    "is_high_unit_price": "High Unit Price Flag (>£50)",
    "is_single_item": "Single Item Order Flag",
    "cust_total_orders": "Customer Order History Count",
    "cust_cancel_rate": "Customer Historical Cancel Rate"
}


def prepare_features(df):
    """
    Extracts and validates features from a dataframe.
    """
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        # If columns are missing, attempt derivation
        dff = df.copy()
        if "items_per_order_ratio" not in dff.columns and "total_quantity" in dff.columns and "n_unique_items" in dff.columns:
            dff["items_per_order_ratio"] = dff["total_quantity"] / (dff["n_unique_items"] + 1e-5)
        if "is_high_unit_price" not in dff.columns and "max_unit_price" in dff.columns:
            dff["is_high_unit_price"] = (dff["max_unit_price"] > 50.0).astype(int)
        if "is_single_item" not in dff.columns and "n_unique_items" in dff.columns:
            dff["is_single_item"] = (dff["n_unique_items"] == 1).astype(int)
        if "avg_item_quantity" not in dff.columns and "total_quantity" in dff.columns and "n_unique_items" in dff.columns:
            dff["avg_item_quantity"] = dff["total_quantity"] / (dff["n_unique_items"] + 1e-5)
        if "avg_item_amount" not in dff.columns and "total_amount" in dff.columns and "n_unique_items" in dff.columns:
            dff["avg_item_amount"] = dff["total_amount"] / (dff["n_unique_items"] + 1e-5)
        for col in FEATURE_COLUMNS:
            if col not in dff.columns:
                dff[col] = 0.0
        return dff[FEATURE_COLUMNS].fillna(0)

    return df[FEATURE_COLUMNS].fillna(0)


def create_sample_order_dict(
    n_unique_items=3,
    total_quantity=10,
    total_amount=150.0,
    avg_unit_price=15.0,
    max_unit_price=25.0,
    hour=14,
    day_of_week=2,
    month=10,
    is_uk=1,
    has_customer_id=1,
    cust_total_orders=5,
    cust_cancel_rate=0.0
):
    """
    Constructs an input dictionary for live prediction.
    """
    avg_item_qty = total_quantity / max(1, n_unique_items)
    avg_item_amt = total_amount / max(1, n_unique_items)
    items_ratio = total_quantity / (n_unique_items + 1e-5)
    is_high_price = 1 if max_unit_price > 50.0 else 0
    is_single = 1 if n_unique_items == 1 else 0

    return {
        "n_unique_items": float(n_unique_items),
        "total_quantity": float(total_quantity),
        "avg_item_quantity": float(avg_item_qty),
        "total_amount": float(total_amount),
        "avg_item_amount": float(avg_item_amt),
        "avg_unit_price": float(avg_unit_price),
        "max_unit_price": float(max_unit_price),
        "hour": int(hour),
        "day_of_week": int(day_of_week),
        "month": int(month),
        "is_uk": int(is_uk),
        "has_customer_id": int(has_customer_id),
        "items_per_order_ratio": float(items_ratio),
        "is_high_unit_price": int(is_high_price),
        "is_single_item": int(is_single),
        "cust_total_orders": float(cust_total_orders),
        "cust_cancel_rate": float(cust_cancel_rate)
    }
