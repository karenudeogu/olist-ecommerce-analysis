"""
rfm.py
------
RFM (Recency · Frequency · Monetary) segmentation module.
"""

import pandas as pd
import numpy as np


# ── Segment label map ────────────────────────────────────────────────────────
SEGMENT_COLORS = {
    "Champions":           "#2ecc71",
    "Loyal Customers":     "#27ae60",
    "Potential Loyalists": "#1abc9c",
    "Recent Customers":    "#3498db",
    "At-Risk":             "#e67e22",
    "Cant Lose Them":      "#e74c3c",
    "Hibernating":         "#95a5a6",
    "Lost":                "#7f8c8d",
}


def compute_rfm(master: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Compute raw RFM metrics from the master order table.

    Parameters
    ----------
    master : output of data_loader.build_master()
    snapshot_date : reference date for recency (defaults to max purchase + 1 day)

    Returns
    -------
    DataFrame indexed by customer_unique_id with columns:
        recency, frequency, monetary
    """
    if snapshot_date is None:
        snapshot_date = master["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = (
        master
        .groupby("customer_unique_id")
        .agg(
            recency=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
            frequency=("order_id", "count"),
            monetary=("payment_value", "sum"),
        )
        .reset_index()
    )
    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Add 1–5 quintile scores for R, F, M and derive segment labels.
    Note: Lower recency = better (scored inversely).
    """
    rfm = rfm.copy()

    rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["m_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

    rfm["rfm_score"] = (
        rfm["r_score"].astype(str)
        + rfm["f_score"].astype(str)
        + rfm["m_score"].astype(str)
    )

    rfm["segment"] = rfm.apply(_assign_segment, axis=1)
    return rfm


def _assign_segment(row: pd.Series) -> str:
    r, f, m = int(row["r_score"]), int(row["f_score"]), int(row["m_score"])
    if r >= 4 and f >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "Recent Customers"
    elif r >= 3 and f <= 2:
        return "Potential Loyalists"
    elif r <= 2 and f >= 3:
        return "At-Risk"
    elif r <= 2 and f <= 2 and m >= 3:
        return "Cant Lose Them"
    elif r == 1 and f == 1:
        return "Lost"
    else:
        return "Hibernating"


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per segment for reporting."""
    summary = (
        rfm.groupby("segment")
        .agg(
            customers=("customer_unique_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    summary["revenue_pct"] = (summary["total_revenue"] / summary["total_revenue"].sum() * 100).round(1)
    return summary
