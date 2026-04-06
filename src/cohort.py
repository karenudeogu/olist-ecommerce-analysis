"""
cohort.py
---------
Monthly cohort retention analysis.
"""

import pandas as pd
import numpy as np


def build_cohort_retention(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a cohort retention matrix.

    Parameters
    ----------
    master : output of data_loader.build_master()

    Returns
    -------
    (cohort_counts, retention_pct)
        cohort_counts : absolute customer counts per cohort × month index
        retention_pct : percentage of initial cohort retained each month
    """
    df = master.copy()

    # Acquisition month = first purchase month per customer
    df["cohort_month"] = (
        df.groupby("customer_unique_id")["order_purchase_timestamp"]
        .transform("min")
        .dt.to_period("M")
    )
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M")
    df["cohort_index"] = (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)

    cohort_data = (
        df.groupby(["cohort_month", "cohort_index"])["customer_unique_id"]
        .nunique()
        .reset_index()
    )

    cohort_counts = cohort_data.pivot(
        index="cohort_month", columns="cohort_index", values="customer_unique_id"
    )

    cohort_size = cohort_counts[0]
    retention_pct = cohort_counts.divide(cohort_size, axis=0).round(4) * 100

    return cohort_counts, retention_pct


def monthly_retention_summary(retention_pct: pd.DataFrame, max_index: int = 12) -> pd.Series:
    """Average retention rate at each month index across all cohorts."""
    cols = [c for c in retention_pct.columns if 0 < c <= max_index]
    return retention_pct[cols].mean().round(2)


def churn_analysis(master: pd.DataFrame, churn_days: int = 180) -> pd.DataFrame:
    """
    Flag customers as churned if their last purchase was > churn_days ago.

    Returns a DataFrame with customer_unique_id, days_since_last_order,
    total_orders, total_spend, is_churned.
    """
    snapshot = master["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    cust = master.groupby("customer_unique_id").agg(
        last_order=("order_purchase_timestamp", "max"),
        total_orders=("order_id", "count"),
        total_spend=("payment_value", "sum"),
    ).reset_index()

    cust["days_since_last_order"] = (snapshot - cust["last_order"]).dt.days
    cust["is_churned"] = (cust["days_since_last_order"] > churn_days).astype(int)

    return cust
