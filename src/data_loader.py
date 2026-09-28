"""
Data loader and preprocessor for Ecommerce Order Cancellation dataset.
Loads raw transaction lines and aggregates them into high-signal order-level records.
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(data_path="data/data.csv"):
    """
    Loads the raw e-commerce transaction dataset.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load with fallback encoding
    try:
        df = pd.read_csv(data_path, encoding="ISO-8859-1", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)

    return df


def clean_and_aggregate_orders(raw_df):
    """
    Cleans raw transactions and transforms them into order-level records.
    """
    df = raw_df.copy()

    # Identify cancellations
    df["InvoiceNo_str"] = df["InvoiceNo"].astype(str)
    df["is_cancelled"] = df["InvoiceNo_str"].str.startswith("C").astype(int)

    # Clean quantities and prices
    df["AbsQuantity"] = df["Quantity"].abs()
    df["UnitPrice"] = df["UnitPrice"].clip(lower=0)
    df["Amount"] = df["AbsQuantity"] * df["UnitPrice"]

    # Parse timestamps
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
    df["Month"] = df["InvoiceDate"].dt.month

    # Aggregate by Invoice
    inv_df = df.groupby("InvoiceNo").agg(
        is_cancelled=("is_cancelled", "first"),
        n_unique_items=("StockCode", "count"),
        total_quantity=("AbsQuantity", "sum"),
        avg_item_quantity=("AbsQuantity", "mean"),
        total_amount=("Amount", "sum"),
        avg_item_amount=("Amount", "mean"),
        avg_unit_price=("UnitPrice", "mean"),
        max_unit_price=("UnitPrice", "max"),
        customer_id=("CustomerID", "first"),
        country=("Country", "first"),
        invoice_date=("InvoiceDate", "first"),
        hour=("Hour", "first"),
        day_of_week=("DayOfWeek", "first"),
        month=("Month", "first")
    ).reset_index()

    # Domain features
    inv_df["is_uk"] = (inv_df["country"] == "United Kingdom").astype(int)
    inv_df["has_customer_id"] = inv_df["customer_id"].notnull().astype(int)
    inv_df["items_per_order_ratio"] = inv_df["total_quantity"] / (inv_df["n_unique_items"] + 1e-5)
    inv_df["is_high_unit_price"] = (inv_df["max_unit_price"] > 50.0).astype(int)
    inv_df["is_single_item"] = (inv_df["n_unique_items"] == 1).astype(int)

    # Customer historical behavior
    cust_known = inv_df[inv_df["has_customer_id"] == 1]
    cust_stats = cust_known.groupby("customer_id").agg(
        cust_total_orders=("InvoiceNo", "count"),
        cust_total_cancellations=("is_cancelled", "sum"),
        cust_avg_order_value=("total_amount", "mean")
    ).reset_index()

    cust_stats["cust_cancel_rate"] = (
        cust_stats["cust_total_cancellations"] / cust_stats["cust_total_orders"]
    ).fillna(0)

    # Merge customer statistics
    inv_df = inv_df.merge(cust_stats, on="customer_id", how="left")
    inv_df["cust_total_orders"] = inv_df["cust_total_orders"].fillna(1)
    inv_df["cust_cancel_rate"] = inv_df["cust_cancel_rate"].fillna(0.0)
    inv_df["cust_avg_order_value"] = inv_df["cust_avg_order_value"].fillna(inv_df["total_amount"])

    return inv_df


def get_or_create_processed_data(raw_path="data/data.csv", cache_path="data/processed_orders.csv"):
    """
    Returns the processed order dataframe, loading from cache if available.
    """
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        return df

    raw_df = load_raw_data(raw_path)
    orders_df = clean_and_aggregate_orders(raw_df)
    orders_df.to_csv(cache_path, index=False)
    return orders_df


if __name__ == "__main__":
    print("Testing data loader...")
    orders = get_or_create_processed_data()
    print(f"Processed orders shape: {orders.shape}")
    print(orders.head())
