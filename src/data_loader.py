"""
data_loader.py
--------------
Loads and merges all Olist datasets into a clean master DataFrame.
"""

import os
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def load_raw() -> dict[str, pd.DataFrame]:
    """Return a dictionary of all raw DataFrames."""
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    return {
        "orders": pd.read_csv(_path("olist_orders_dataset.csv"), parse_dates=date_cols),
        "customers": pd.read_csv(_path("olist_customers_dataset.csv")),
        "order_items": pd.read_csv(_path("olist_order_items_dataset.csv")),
        "payments": pd.read_csv(_path("olist_order_payments_dataset.csv")),
        "reviews": pd.read_csv(_path("olist_order_reviews_dataset.csv")),
        "products": pd.read_csv(_path("olist_products_dataset.csv")),
        "sellers": pd.read_csv(_path("olist_sellers_dataset.csv")),
        "category_translation": pd.read_csv(_path("product_category_name_translation.csv")),
    }


def build_master(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Joins orders → customers → payments (aggregated) for delivered orders only.

    Returns
    -------
    pd.DataFrame with columns:
        order_id, customer_id, customer_unique_id, customer_state,
        order_purchase_timestamp, order_delivered_customer_date,
        order_estimated_delivery_date, payment_value, delivery_days,
        is_late_delivery
    """
    orders_delivered = raw["orders"][raw["orders"]["order_status"] == "delivered"].copy()

    payments_agg = (
        raw["payments"]
        .groupby("order_id")["payment_value"]
        .sum()
        .reset_index()
    )

    master = (
        orders_delivered
        .merge(raw["customers"], on="customer_id")
        .merge(payments_agg, on="order_id")
    )

    master["delivery_days"] = (
        master["order_delivered_customer_date"] - master["order_purchase_timestamp"]
    ).dt.days

    master["is_late_delivery"] = (
        master["order_delivered_customer_date"] > master["order_estimated_delivery_date"]
    ).astype(int)

    return master


def build_items_master(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Items enriched with English category names and order dates."""
    products_en = raw["products"].merge(
        raw["category_translation"], on="product_category_name", how="left"
    )
    orders_delivered = raw["orders"][raw["orders"]["order_status"] == "delivered"][
        ["order_id", "order_purchase_timestamp"]
    ]
    items = (
        raw["order_items"]
        .merge(products_en[["product_id", "product_category_name_english"]], on="product_id")
        .merge(orders_delivered, on="order_id")
    )
    return items
