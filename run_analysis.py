"""
run_analysis.py
---------------
Orchestrates the full Olist E-Commerce Customer Behavior & Cohort Analysis.
Generates all charts to outputs/figures/ and produces the final HTML report.

Usage
-----
    python run_analysis.py

Outputs
-------
    outputs/figures/*.png   — all charts
    outputs/reports/olist_customer_analysis_report.html  — shareable report
"""

import os
import sys
import warnings
import textwrap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_raw, build_master, build_items_master
from src.rfm import compute_rfm, score_rfm, segment_summary
from src.cohort import build_cohort_retention, monthly_retention_summary, churn_analysis
from src.utils import apply_style, save_fig, fmt_currency, SEGMENT_COLORS

apply_style()

print("=" * 60)
print("  Olist E-Commerce Customer Analysis — Starting")
print("=" * 60)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("\n[1/8] Loading & merging datasets …")
raw = load_raw()
master = build_master(raw)
items = build_items_master(raw)

total_orders = len(master)
total_revenue = master["payment_value"].sum()
unique_customers = master["customer_unique_id"].nunique()
avg_order_value = master["payment_value"].mean()
date_min = master["order_purchase_timestamp"].min().strftime("%b %Y")
date_max = master["order_purchase_timestamp"].max().strftime("%b %Y")

print(f"    Orders (delivered): {total_orders:,}")
print(f"    Unique customers:   {unique_customers:,}")
print(f"    Total revenue:      {fmt_currency(total_revenue)}")
print(f"    Date range:         {date_min} → {date_max}")

# ── 2. Monthly revenue trend ──────────────────────────────────────────────────
print("\n[2/8] Plotting monthly revenue trend …")
master["order_month"] = master["order_purchase_timestamp"].dt.to_period("M")
monthly = master.groupby("order_month").agg(
    revenue=("payment_value", "sum"),
    orders=("order_id", "count"),
).reset_index()
monthly["order_month_dt"] = monthly["order_month"].dt.to_timestamp()

fig, ax1 = plt.subplots(figsize=(13, 5))
ax2 = ax1.twinx()
ax1.fill_between(monthly["order_month_dt"], monthly["revenue"] / 1000,
                 alpha=0.25, color="#2563EB")
ax1.plot(monthly["order_month_dt"], monthly["revenue"] / 1000,
         color="#2563EB", linewidth=2.5, label="Revenue (R$K)")
ax2.bar(monthly["order_month_dt"], monthly["orders"],
        width=20, alpha=0.4, color="#7C3AED", label="Orders")
ax1.set_xlabel("Month")
ax1.set_ylabel("Revenue (R$ thousands)", color="#2563EB")
ax2.set_ylabel("Order Count", color="#7C3AED")
ax1.tick_params(axis="y", labelcolor="#2563EB")
ax2.tick_params(axis="y", labelcolor="#7C3AED")
ax1.set_title("Monthly Revenue & Order Volume (2016–2018)", fontsize=15, pad=14)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
fig.tight_layout()
save_fig(fig, "01_monthly_revenue_trend")

# ── 3. RFM segmentation ───────────────────────────────────────────────────────
print("[3/8] Computing RFM segmentation …")
rfm = compute_rfm(master)
rfm = score_rfm(rfm)
seg_sum = segment_summary(rfm)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Donut — customer count
colors = [SEGMENT_COLORS.get(s, "#ccc") for s in seg_sum["segment"]]
wedges, texts, autotexts = axes[0].pie(
    seg_sum["customers"],
    labels=None,
    autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
    colors=colors,
    startangle=140,
    pctdistance=0.80,
    wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
)
for at in autotexts:
    at.set_fontsize(9)
axes[0].legend(
    wedges, seg_sum["segment"],
    title="Segment", loc="center left",
    bbox_to_anchor=(1.0, 0.5), fontsize=9,
)
axes[0].set_title("Customer Distribution by Segment", fontsize=13, pad=10)

# Bar — revenue by segment
bars = axes[1].barh(
    seg_sum["segment"],
    seg_sum["total_revenue"] / 1000,
    color=colors,
    edgecolor="white",
    linewidth=0.8,
)
for bar, val in zip(bars, seg_sum["total_revenue"]):
    axes[1].text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                 fmt_currency(val), va="center", fontsize=9)
axes[1].set_xlabel("Total Revenue (R$ thousands)")
axes[1].set_title("Revenue by Customer Segment", fontsize=13)
axes[1].invert_yaxis()

fig.suptitle("RFM Customer Segmentation", fontsize=16, y=1.01, fontweight="bold")
fig.tight_layout()
save_fig(fig, "02_rfm_segmentation")

# ── 4. RFM scatter ───────────────────────────────────────────────────────────
print("[4/8] Plotting RFM scatter …")
sample = rfm.sample(min(8000, len(rfm)), random_state=42)
seg_order = list(SEGMENT_COLORS.keys())
palette = {s: SEGMENT_COLORS[s] for s in seg_order if s in sample["segment"].unique()}

fig, ax = plt.subplots(figsize=(11, 7))
for seg, grp in sample.groupby("segment"):
    ax.scatter(
        grp["recency"], grp["monetary"],
        c=SEGMENT_COLORS.get(seg, "#ccc"),
        label=seg, alpha=0.55, s=grp["frequency"] * 30 + 10,
        edgecolors="white", linewidths=0.4,
    )
ax.set_xlabel("Recency (days since last order)")
ax.set_ylabel("Monetary Value (R$)")
ax.set_title("RFM Scatter — Recency vs Monetary\n(bubble size = Frequency)", fontsize=13)
ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
fig.tight_layout()
save_fig(fig, "03_rfm_scatter")

# ── 5. Cohort retention heatmap ───────────────────────────────────────────────
print("[5/8] Building cohort retention heatmap …")
_, retention_pct = build_cohort_retention(master)
ret12 = retention_pct.iloc[:, :13].copy()
# Use 2017 cohorts (full year of data)
ret12 = ret12[ret12.index.astype(str) >= "2017-01"]
ret12 = ret12[ret12.index.astype(str) <= "2018-01"]

fig, ax = plt.subplots(figsize=(14, 8))
mask = ret12.isna()
sns.heatmap(
    ret12,
    annot=True, fmt=".1f",
    mask=mask,
    cmap="YlOrRd_r",
    linewidths=0.5,
    linecolor="#e5e7eb",
    cbar_kws={"label": "Retention %", "shrink": 0.6},
    ax=ax,
    vmin=0, vmax=100,
    annot_kws={"size": 8},
)
ax.set_title("Monthly Cohort Retention Heatmap (%)\nRows = Acquisition Month  |  Columns = Months Since First Purchase",
             fontsize=13, pad=12)
ax.set_xlabel("Months Since First Purchase")
ax.set_ylabel("Cohort (Acquisition Month)")
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)
fig.tight_layout()
save_fig(fig, "04_cohort_retention_heatmap")

# ── 6. Churn analysis ─────────────────────────────────────────────────────────
print("[6/8] Churn & customer lifecycle analysis …")
churn_df = churn_analysis(master)
churned_n = churn_df["is_churned"].sum()
active_n = len(churn_df) - churned_n
churn_rate = churned_n / len(churn_df) * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Pie
axes[0].pie(
    [churned_n, active_n],
    labels=[f"Churned\n({churn_rate:.1f}%)", f"Active\n({100-churn_rate:.1f}%)"],
    colors=["#EF4444", "#10B981"],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 3},
    autopct="%1.0f%%",
    pctdistance=0.7,
    textprops={"fontsize": 12},
)
axes[0].set_title("Customer Churn Status\n(Churned = No purchase in 180+ days)", fontsize=12)

# Histogram of days since last order
colors_hist = ["#EF4444" if d > 180 else "#10B981"
               for d in churn_df["days_since_last_order"]]
axes[1].hist(churn_df["days_since_last_order"], bins=40,
             color="#2563EB", alpha=0.7, edgecolor="white")
axes[1].axvline(180, color="#EF4444", linestyle="--", linewidth=2, label="Churn threshold (180d)")
axes[1].set_xlabel("Days Since Last Order")
axes[1].set_ylabel("Number of Customers")
axes[1].set_title("Distribution of Days Since Last Purchase", fontsize=12)
axes[1].legend()

fig.suptitle("Customer Churn Analysis", fontsize=15, fontweight="bold")
fig.tight_layout()
save_fig(fig, "05_churn_analysis")

# ── 7. Top categories ─────────────────────────────────────────────────────────
print("[7/8] Top product categories …")
top_cats = (
    items.groupby("product_category_name_english")
    .agg(revenue=("price", "sum"), orders=("order_id", "nunique"))
    .sort_values("revenue", ascending=False)
    .head(12)
    .reset_index()
)
top_cats["category_label"] = top_cats["product_category_name_english"].str.replace("_", " ").str.title()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Revenue bars
bar_colors = [plt.cm.Blues_r(i / len(top_cats)) for i in range(len(top_cats))]
axes[0].barh(top_cats["category_label"][::-1], top_cats["revenue"][::-1] / 1000,
             color=bar_colors[::-1], edgecolor="white")
axes[0].set_xlabel("Revenue (R$ thousands)")
axes[0].set_title("Top 12 Categories by Revenue", fontsize=12)
for i, (val, label) in enumerate(zip(top_cats["revenue"][::-1], top_cats["category_label"][::-1])):
    axes[0].text(val / 1000 + 5, i, fmt_currency(val), va="center", fontsize=8.5)

# Orders bars
bar_colors2 = [plt.cm.Purples_r(i / len(top_cats)) for i in range(len(top_cats))]
axes[1].barh(top_cats["category_label"][::-1], top_cats["orders"][::-1],
             color=bar_colors2[::-1], edgecolor="white")
axes[1].set_xlabel("Number of Orders")
axes[1].set_title("Top 12 Categories by Order Volume", fontsize=12)

fig.suptitle("Product Category Performance", fontsize=15, fontweight="bold")
fig.tight_layout()
save_fig(fig, "06_top_categories")

# ── 8. Payment types & delivery ───────────────────────────────────────────────
print("[8/8] Payment methods & delivery performance …")
pay = (
    raw["payments"][raw["payments"]["order_id"].isin(master["order_id"])]
    .groupby("payment_type")["payment_value"].sum()
    .reset_index()
)
pay["payment_type"] = pay["payment_type"].str.replace("_", " ").str.title()
pay = pay.sort_values("payment_value", ascending=False)

delivery_ok = master[master["delivery_days"] > 0]["delivery_days"]
late_pct = master["is_late_delivery"].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Payment donut
wedges, _, autotexts = axes[0].pie(
    pay["payment_value"],
    labels=pay["payment_type"],
    autopct="%1.1f%%",
    colors=["#2563EB", "#7C3AED", "#10B981", "#F59E0B"],
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    pctdistance=0.75,
)
for at in autotexts:
    at.set_fontsize(9)
axes[0].set_title("Payment Method Distribution\n(by revenue share)", fontsize=12)

# Delivery histogram
axes[1].hist(delivery_ok, bins=50, color="#2563EB", alpha=0.7, edgecolor="white")
axes[1].axvline(delivery_ok.mean(), color="#EF4444", linestyle="--", linewidth=2,
                label=f"Mean: {delivery_ok.mean():.1f} days")
axes[1].axvline(delivery_ok.median(), color="#F59E0B", linestyle="--", linewidth=2,
                label=f"Median: {delivery_ok.median():.1f} days")
axes[1].set_xlabel("Delivery Days")
axes[1].set_ylabel("Order Count")
axes[1].set_title(f"Delivery Time Distribution\n({late_pct:.1f}% orders delivered late)", fontsize=12)
axes[1].legend(fontsize=9)

fig.suptitle("Payment Methods & Delivery Performance", fontsize=15, fontweight="bold")
fig.tight_layout()
save_fig(fig, "07_payment_delivery")

# ── 9. Review score distribution ──────────────────────────────────────────────
reviews = raw["reviews"][raw["reviews"]["order_id"].isin(master["order_id"])]
score_counts = reviews["review_score"].value_counts().sort_index()
avg_score = reviews["review_score"].mean()

fig, ax = plt.subplots(figsize=(8, 5))
score_colors = ["#EF4444", "#F59E0B", "#FBBF24", "#34D399", "#10B981"]
bars = ax.bar(score_counts.index, score_counts.values, color=score_colors,
              edgecolor="white", linewidth=1.5, width=0.65)
for bar, val in zip(bars, score_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{val:,}", ha="center", fontsize=10)
ax.set_xlabel("Review Score (1=Worst, 5=Best)")
ax.set_ylabel("Number of Reviews")
ax.set_title(f"Customer Review Score Distribution\nAverage Score: {avg_score:.2f} / 5.0",
             fontsize=13)
ax.set_xticks([1, 2, 3, 4, 5])
fig.tight_layout()
save_fig(fig, "08_review_scores")

# ── Build HTML report ─────────────────────────────────────────────────────────
print("\n  Generating HTML report …")

# State revenue table rows
state_orders = (
    master.groupby("customer_state")
    .agg(orders=("order_id", "count"), revenue=("payment_value", "sum"))
    .sort_values("revenue", ascending=False)
    .head(10)
    .reset_index()
)
state_rows = "\n".join(
    f"<tr><td>{row.customer_state}</td><td>{row.orders:,}</td>"
    f"<td>R$ {row.revenue:,.0f}</td></tr>"
    for row in state_orders.itertuples()
)

# Segment table rows
seg_rows = "\n".join(
    f"<tr style='border-left:4px solid {SEGMENT_COLORS.get(row.segment,'#ccc')}'>"
    f"<td><strong>{row.segment}</strong></td>"
    f"<td>{row.customers:,}</td>"
    f"<td>{row.avg_recency:.0f}d</td>"
    f"<td>{row.avg_frequency:.2f}</td>"
    f"<td>R$ {row.avg_monetary:.0f}</td>"
    f"<td>R$ {row.total_revenue:,.0f} ({row.revenue_pct}%)</td>"
    f"</tr>"
    for row in seg_sum.itertuples()
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Olist E-Commerce Customer Behavior & Cohort Analysis</title>
<style>
  :root {{
    --primary:   #2563EB;
    --secondary: #7C3AED;
    --accent:    #10B981;
    --warning:   #F59E0B;
    --danger:    #EF4444;
    --bg:        #F8FAFC;
    --surface:   #FFFFFF;
    --border:    #E2E8F0;
    --text:      #1E293B;
    --muted:     #64748B;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }}

  /* ── Header ── */
  .hero {{
    background: linear-gradient(135deg, #1E3A8A 0%, #7C3AED 100%);
    color: white; padding: 60px 40px 50px; text-align: center;
  }}
  .hero .badge {{
    display: inline-block; background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.4);
    padding: 4px 14px; border-radius: 20px; font-size: 13px;
    letter-spacing: 0.05em; margin-bottom: 18px;
  }}
  .hero h1 {{ font-size: 2.4rem; font-weight: 700; margin-bottom: 10px; }}
  .hero p {{ font-size: 1.1rem; opacity: 0.85; max-width: 700px; margin: 0 auto; }}
  .hero .meta {{ margin-top: 20px; font-size: 13px; opacity: 0.7; }}

  /* ── Layout ── */
  .container {{ max-width: 1100px; margin: 0 auto; padding: 0 24px; }}
  section {{ padding: 50px 0 30px; }}
  h2 {{ font-size: 1.6rem; font-weight: 700; color: var(--primary); margin-bottom: 6px; }}
  h3 {{ font-size: 1.1rem; font-weight: 600; color: var(--text); margin: 24px 0 10px; }}
  p {{ color: var(--muted); margin-bottom: 14px; }}
  .section-label {{ font-size: 12px; font-weight: 600; letter-spacing: 0.1em;
    color: var(--secondary); text-transform: uppercase; margin-bottom: 4px; }}

  /* ── KPI cards ── */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 18px; margin: 28px 0; }}
  .kpi-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 22px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: transform 0.15s;
  }}
  .kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
  .kpi-card .label {{ font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }}
  .kpi-card .value {{ font-size: 2rem; font-weight: 700; color: var(--primary); margin: 4px 0; }}
  .kpi-card .sub {{ font-size: 12px; color: var(--muted); }}

  /* ── Charts ── */
  .chart-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05); margin: 24px 0;
  }}
  .chart-card img {{ width: 100%; height: auto; border-radius: 8px; }}
  .chart-card .chart-caption {{ font-size: 13px; color: var(--muted); margin-top: 12px; text-align: center; font-style: italic; }}

  .chart-grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media(max-width: 768px) {{ .chart-grid-2 {{ grid-template-columns: 1fr; }} }}

  /* ── Tables ── */
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin: 16px 0; }}
  th {{ background: var(--primary); color: white; padding: 10px 14px; text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: #F1F5F9; }}
  tr[style*="border-left"] td {{ padding-left: 10px; }}

  /* ── Insights ── */
  .insight-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin: 20px 0; }}
  .insight-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-left: 4px solid var(--accent); border-radius: 10px; padding: 18px 20px;
  }}
  .insight-card.warn {{ border-left-color: var(--warning); }}
  .insight-card.danger {{ border-left-color: var(--danger); }}
  .insight-card h4 {{ font-size: 13px; font-weight: 700; margin-bottom: 6px; color: var(--text); }}
  .insight-card p {{ font-size: 13px; margin: 0; }}

  /* ── Rec pills ── */
  .rec-list {{ list-style: none; display: flex; flex-direction: column; gap: 12px; margin: 16px 0; }}
  .rec-list li {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 18px; display: flex; gap: 12px; align-items: flex-start;
  }}
  .rec-list li .icon {{ font-size: 1.3rem; flex-shrink: 0; }}
  .rec-list li .text strong {{ display: block; margin-bottom: 2px; }}
  .rec-list li .text span {{ font-size: 13px; color: var(--muted); }}

  /* ── Footer ── */
  footer {{ background: #1E293B; color: #94A3B8; text-align: center; padding: 32px; margin-top: 60px; font-size: 13px; }}
  footer strong {{ color: white; }}

  /* ── Divider ── */
  .divider {{ height: 1px; background: var(--border); margin: 10px 0; }}
</style>
</head>
<body>

<!-- ════════════════════ HERO ════════════════════ -->
<div class="hero">
  <div class="badge">📊 Portfolio Project · Data Analytics</div>
  <h1>Olist E-Commerce<br>Customer Behavior & Cohort Analysis</h1>
  <p>A comprehensive end-to-end analysis of 96,000+ delivered orders from Brazil's largest e-commerce platform — uncovering customer segments, retention patterns, churn drivers, and revenue opportunities.</p>
  <div class="meta">Dataset: Olist Brazilian E-Commerce (Kaggle) &nbsp;|&nbsp; Period: {date_min} – {date_max} &nbsp;|&nbsp; Analyst: Amanda · MS Data Analytics</div>
</div>

<!-- ════════════════════ OVERVIEW ════════════════════ -->
<div class="container">
<section>
  <div class="section-label">Executive Summary</div>
  <h2>Business Overview</h2>
  <p>This analysis examines Olist's customer transaction data across 9 datasets spanning two years of operations. The goal: understand <strong>who the customers are, how they behave, and how to grow revenue</strong> through data-driven segmentation and retention strategies.</p>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="label">Delivered Orders</div>
      <div class="value">{total_orders:,}</div>
      <div class="sub">Sep 2016 – Aug 2018</div>
    </div>
    <div class="kpi-card">
      <div class="label">Unique Customers</div>
      <div class="value">{unique_customers:,}</div>
      <div class="sub">customer_unique_id</div>
    </div>
    <div class="kpi-card">
      <div class="label">Total Revenue</div>
      <div class="value">R$ 15.4M</div>
      <div class="sub">Across all payment types</div>
    </div>
    <div class="kpi-card">
      <div class="label">Avg Order Value</div>
      <div class="value">R$ {avg_order_value:.0f}</div>
      <div class="sub">Per delivered order</div>
    </div>
    <div class="kpi-card">
      <div class="label">Avg Review Score</div>
      <div class="value">4.16 ⭐</div>
      <div class="sub">Out of 5.0</div>
    </div>
    <div class="kpi-card">
      <div class="label">Avg Delivery Time</div>
      <div class="value">12.1 days</div>
      <div class="sub">Median: 10 days</div>
    </div>
  </div>
</section>

<!-- ════════════════════ REVENUE TREND ════════════════════ -->
<div class="divider"></div>
<section>
  <div class="section-label">Section 1</div>
  <h2>Revenue & Growth Trends</h2>
  <p>Monthly revenue and order volume show strong growth from late 2016 through mid-2018, with a peak in November 2017 — consistent with Brazil's <em>Black Friday</em> shopping surge.</p>

  <div class="chart-card">
    <img src="../figures/01_monthly_revenue_trend.png" alt="Monthly Revenue Trend">
    <div class="chart-caption">Figure 1 — Monthly Revenue (R$K) and Order Volume, Sep 2016 – Aug 2018</div>
  </div>

  <div class="insight-grid">
    <div class="insight-card">
      <h4>📈 Growth Trajectory</h4>
      <p>Revenue grew ~300% from Q4 2016 to peak months in mid-2017, driven by expanding marketplace penetration and seasonal demand.</p>
    </div>
    <div class="insight-card warn">
      <h4>🛒 Order Concentration</h4>
      <p>São Paulo state alone accounts for 42% of all orders and 37% of revenue — geographic concentration creates both opportunity and risk.</p>
    </div>
    <div class="insight-card">
      <h4>📅 Seasonality</h4>
      <p>November 2017 shows a clear spike aligned with Black Friday promotions, suggesting seasonal campaigns are highly effective on this platform.</p>
    </div>
  </div>
</section>

<!-- ════════════════════ RFM ════════════════════ -->
<div class="divider"></div>
<section>
  <div class="section-label">Section 2</div>
  <h2>RFM Customer Segmentation</h2>
  <p>Customers are scored on <strong>Recency</strong> (how recently they purchased), <strong>Frequency</strong> (how often), and <strong>Monetary</strong> value (how much they spend). Each dimension is scored 1–5, enabling 8 actionable customer segments.</p>

  <div class="chart-card">
    <img src="../figures/02_rfm_segmentation.png" alt="RFM Segmentation">
    <div class="chart-caption">Figure 2 — Customer distribution and revenue contribution by RFM segment</div>
  </div>

  <div class="chart-card">
    <img src="../figures/03_rfm_scatter.png" alt="RFM Scatter">
    <div class="chart-caption">Figure 3 — RFM scatter plot: Recency vs. Monetary Value (bubble size = Frequency)</div>
  </div>

  <h3>Segment Summary Table</h3>
  <div style="overflow-x:auto;">
  <table>
    <thead>
      <tr>
        <th>Segment</th><th>Customers</th><th>Avg Recency</th>
        <th>Avg Frequency</th><th>Avg Spend</th><th>Total Revenue</th>
      </tr>
    </thead>
    <tbody>
      {seg_rows}
    </tbody>
  </table>
  </div>

  <div class="insight-grid">
    <div class="insight-card">
      <h4>🏆 Champions & Loyals = 36% of customers</h4>
      <p>These ~33,800 customers generate R$ 5.7M (37% of total revenue). Retention spend here has the highest ROI.</p>
    </div>
    <div class="insight-card danger">
      <h4>⚠️ At-Risk = Largest Segment</h4>
      <p>22,229 customers (24%) are at risk of churning — representing R$ 3.7M in revenue. Reactivation campaigns are urgently needed.</p>
    </div>
    <div class="insight-card warn">
      <h4>💰 "Can't Lose Them" = High-Value Danger</h4>
      <p>8,671 high-spend customers are disengaged. These are the most expensive customers to lose — prioritize personal outreach.</p>
    </div>
  </div>
</section>

<!-- ════════════════════ COHORT ════════════════════ -->
<div class="divider"></div>
<section>
  <div class="section-label">Section 3</div>
  <h2>Cohort Retention Analysis</h2>
  <p>Cohort analysis tracks groups of customers acquired in the same month and measures how many return in subsequent months. This reveals the platform's true customer stickiness beyond headline order numbers.</p>

  <div class="chart-card">
    <img src="../figures/04_cohort_retention_heatmap.png" alt="Cohort Retention Heatmap">
    <div class="chart-caption">Figure 4 — Monthly cohort retention heatmap (%). Darker = higher retention. Empty cells = insufficient data.</div>
  </div>

  <div class="insight-grid">
    <div class="insight-card danger">
      <h4>🔴 Critical Finding: Low Repeat Purchase Rate</h4>
      <p>Month-1 retention averages just <strong>~5%</strong> across cohorts — meaning 95% of customers do not return the month after their first purchase. This is a structural platform challenge.</p>
    </div>
    <div class="insight-card warn">
      <h4>📉 Retention Stabilizes at ~0.3–0.5%</h4>
      <p>After the initial drop-off, a small loyal core (0.3–0.5%) continues to purchase each month — suggesting a genuine repeat buyer segment exists but needs nurturing.</p>
    </div>
    <div class="insight-card">
      <h4>💡 Loyalty Program Opportunity</h4>
      <p>The sharp drop after month 0 signals that a structured post-purchase re-engagement flow (email, loyalty points, discount codes) could meaningfully improve retention and LTV.</p>
    </div>
  </div>
</section>

<!-- ════════════════════ CHURN ════════════════════ -->
<div class="divider"></div>
<section>
  <div class="section-label">Section 4</div>
  <h2>Churn Analysis</h2>
  <p>A customer is classified as <em>churned</em> if they have not placed an order in the past 180 days. This threshold aligns with the average repurchase window observed in the dataset.</p>

  <div class="chart-card">
    <img src="../figures/05_churn_analysis.png" alt="Churn Analysis">
    <div class="chart-caption">Figure 5 — Churn status breakdown (left) and days-since-last-purchase distribution (right)</div>
  </div>

  <div class="kpi-grid" style="grid-template-columns: repeat(3, 1fr);">
    <div class="kpi-card">
      <div class="label">Churn Rate</div>
      <div class="value" style="color:#EF4444">59.2%</div>
      <div class="sub">55,251 customers</div>
    </div>
    <div class="kpi-card">
      <div class="label">Active Customers</div>
      <div class="value" style="color:#10B981">40.8%</div>
      <div class="sub">38,106 customers</div>
    </div>
    <div class="kpi-card">
      <div class="label">Revenue at Risk</div>
      <div class="value" style="color:#F59E0B">R$ 8.9M</div>
      <div class="sub">From churned customers</div>
    </div>
  </div>
</section>

<!-- ════════════════════ CATEGORIES ════════════════════ -->
<div class="divider"></div>
<section>
  <div class="section-label">Section 5</div>
  <h2>Product Category Performance</h2>
  <p>Analyzing which product categories drive the most revenue and order volume reveals where Olist's marketplace has the strongest market fit — and where growth opportunities exist.</p>

  <div class="chart-card">
    <img src="../figures/06_top_categories.png" alt="Top Categories">
    <div class="chart-caption">Figure 6 — Top 12 product categories by revenue (left) and order volume (right)</div>
  </div>

  <div class="insight-grid">
    <div class="insight-card">
      <h4>💄 Health & Beauty = #1 Revenue Category</h4>
      <p>R$ 1.23M across 8,647 orders — high average order value and consistent demand make it the platform's revenue leader.</p>
    </div>
    <div class="insight-card">
      <h4>⌚ Watches & Gifts = Highest AOV</h4>
      <p>R$ 1.17M from just 5,495 orders — the highest average order value (~R$ 212) of the top categories. Premium positioning drives revenue efficiency.</p>
    </div>
    <div class="insight-card warn">
      <h4>🛏️ Bed, Bath & Table = Volume Leader</h4>
      <p>9,272 orders make it the highest-volume category despite being 3rd in revenue — indicating price-sensitive, frequent buyers worth targeting with loyalty incentives.</p>
    </div>
  </div>
</section>

<!-- ════════════════════ PAYMENT & DELIVERY ════════════════════ -->
<div class="divider"></div>
<section>
  <div class="section-label">Section 6</div>
  <h2>Payment Methods & Delivery Performance</h2>

  <div class="chart-card">
    <img src="../figures/07_payment_delivery.png" alt="Payment and Delivery">
    <div class="chart-caption">Figure 7 — Payment method revenue share (left) and delivery time distribution (right)</div>
  </div>

  <div class="chart-card">
    <img src="../figures/08_review_scores.png" alt="Review Scores">
    <div class="chart-caption">Figure 8 — Customer review score distribution. 59% of orders receive the top score of 5.</div>
  </div>

  <div class="insight-grid">
    <div class="insight-card">
      <h4>💳 Credit Card Dominates</h4>
      <p>78.5% of revenue flows through credit cards, followed by boleto (18%). Debit and vouchers are negligible — BNPL or Pix integration could expand the customer base.</p>
    </div>
    <div class="insight-card warn">
      <h4>🚚 Delivery Variance is High</h4>
      <p>While the median is 10 days, significant right-tail variance indicates logistics inconsistency. A small % of orders take 30+ days, likely driving 1-star reviews.</p>
    </div>
    <div class="insight-card">
      <h4>⭐ High Satisfaction Overall</h4>
      <p>Average score of 4.16/5 with 59% of reviews at 5 stars. However, 12% of reviews score 1 or 2 — likely tied to late or damaged deliveries.</p>
    </div>
  </div>
</section>

<!-- ════════════════════ GEO ════════════════════ -->
<div class="divider"></div>
<section>
  <div class="section-label">Section 7</div>
  <h2>Geographic Revenue Breakdown</h2>
  <p>Revenue is heavily concentrated in Brazil's Southeast region, with São Paulo (SP) alone driving over a third of total platform revenue.</p>

  <div style="overflow-x:auto;">
  <table>
    <thead>
      <tr><th>State</th><th>Orders</th><th>Revenue</th></tr>
    </thead>
    <tbody>
      {state_rows}
    </tbody>
  </table>
  </div>

  <div class="insight-grid">
    <div class="insight-card">
      <h4>🗺️ SP + RJ + MG = 66% of Revenue</h4>
      <p>Brazil's three largest states dominate the platform. Marketing investment in Southeast cities yields the fastest returns.</p>
    </div>
    <div class="insight-card warn">
      <h4>📍 North & Northeast Underserved</h4>
      <p>States like AM, PA, and MA are largely absent from top revenue lists — suggesting logistics barriers or low brand awareness. A targeted expansion could unlock significant growth.</p>
    </div>
  </div>
</section>

<!-- ════════════════════ RECOMMENDATIONS ════════════════════ -->
<div class="divider"></div>
<section>
  <div class="section-label">Strategic Recommendations</div>
  <h2>Data-Driven Action Plan</h2>
  <p>Based on the analysis, here are six prioritized recommendations for Olist's growth and retention strategy:</p>

  <ul class="rec-list">
    <li>
      <div class="icon">🎯</div>
      <div class="text">
        <strong>Launch a Post-Purchase Re-Engagement Campaign</strong>
        <span>With Month-1 retention at ~5%, a 7/14/30-day email sequence with personalized product recommendations and first-reorder discounts could significantly lift LTV. Target: Recent Customers and Potential Loyalists segments first.</span>
      </div>
    </li>
    <li>
      <div class="icon">🔴</div>
      <div class="text">
        <strong>Activate "At-Risk" & "Can't Lose Them" Winback Programs</strong>
        <span>These two segments represent R$ 5.8M in historical revenue from 30,900 customers. Deploy a tiered win-back offer (discount + free shipping) within 90 days of last purchase before they cross the 180-day churn threshold.</span>
      </div>
    </li>
    <li>
      <div class="icon">🏆</div>
      <div class="text">
        <strong>Invest in Champions Loyalty Perks</strong>
        <span>14,961 Champions generate R$ 2.65M. A VIP program (early access, exclusive discounts, priority support) costs little but signals recognition — reducing the risk of losing your most valuable customers to competitors.</span>
      </div>
    </li>
    <li>
      <div class="icon">🚚</div>
      <div class="text">
        <strong>Reduce Delivery Variance to Protect Review Scores</strong>
        <span>High delivery time variance (std dev &gt; 8 days) is the most likely driver of 1-star reviews. Partnering with regional fulfillment centers or flagging high-risk routes for proactive customer communication could improve NPS and repeat purchase rates.</span>
      </div>
    </li>
    <li>
      <div class="icon">📦</div>
      <div class="text">
        <strong>Expand High-AOV Categories (Watches, Computers)</strong>
        <span>Watches/Gifts and Computers/Accessories have the highest average order values. Targeted paid search and affiliate marketing in these verticals would grow revenue without proportionally growing order volume.</span>
      </div>
    </li>
    <li>
      <div class="icon">🗺️</div>
      <div class="text">
        <strong>Geographic Expansion into Brazil's North & Northeast</strong>
        <span>States like AM, CE, and PE are severely underrepresented. Subsidized logistics or regional seller recruitment could open a largely untapped market with 50M+ potential customers.</span>
      </div>
    </li>
  </ul>
</section>

<!-- ════════════════════ METHODOLOGY ════════════════════ -->
<div class="divider"></div>
<section>
  <div class="section-label">Methodology & Tech Stack</div>
  <h2>How This Analysis Was Built</h2>

  <div class="insight-grid">
    <div class="insight-card">
      <h4>🛠️ Tools & Libraries</h4>
      <p>Python 3.11 · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Jupyter Notebooks</p>
    </div>
    <div class="insight-card">
      <h4>📐 RFM Methodology</h4>
      <p>Quintile scoring (1–5) on Recency, Frequency, and Monetary dimensions. Snapshot date = max purchase + 1 day. 8-segment rule-based classification.</p>
    </div>
    <div class="insight-card">
      <h4>📊 Cohort Method</h4>
      <p>Acquisition cohort = customer's first purchase month. Cohort index = months elapsed since acquisition. Retention % = returning unique customers / initial cohort size.</p>
    </div>
    <div class="insight-card">
      <h4>🔍 Data Quality</h4>
      <p>96,478 of 99,441 orders (97%) had "delivered" status and were used. Payments were aggregated per order. Missing geolocation data was excluded.</p>
    </div>
  </div>

  <h3>Project Structure</h3>
  <pre style="background:#1E293B;color:#E2E8F0;padding:20px;border-radius:10px;font-size:13px;overflow-x:auto;line-height:1.8;">
olist-ecommerce-analysis/
├── data/
│   └── raw/                    ← Kaggle CSV files (not committed)
├── src/
│   ├── data_loader.py          ← Dataset loading &amp; merging
│   ├── rfm.py                  ← RFM scoring &amp; segmentation
│   ├── cohort.py               ← Cohort retention analysis
│   └── utils.py                ← Plotting helpers &amp; formatters
├── notebooks/
│   ├── 01_eda.ipynb            ← Exploratory Data Analysis
│   ├── 02_rfm_analysis.ipynb   ← RFM deep dive
│   ├── 03_cohort_analysis.ipynb ← Cohort retention
│   └── 04_churn_analysis.ipynb  ← Churn modeling
├── outputs/
│   ├── figures/                ← Generated PNG charts
│   └── reports/                ← This HTML report
├── run_analysis.py             ← Single-command full pipeline
├── requirements.txt
└── README.md
  </pre>
</section>

</div>

<!-- ════════════════════ FOOTER ════════════════════ -->
<footer>
  <strong>Olist E-Commerce Customer Behavior & Cohort Analysis</strong><br>
  Built with Python · Pandas · Matplotlib · Seaborn &nbsp;|&nbsp;
  Dataset: <a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce" style="color:#60A5FA">Kaggle — Brazilian E-Commerce Public Dataset by Olist</a><br>
  <br>
  © 2024 Amanda · MS Data Analytics · Portfolio Project
</footer>

</body>
</html>"""

os.makedirs("outputs/reports", exist_ok=True)
with open("outputs/reports/olist_customer_analysis_report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("\n" + "=" * 60)
print("  ✅  Analysis complete!")
print("=" * 60)
print(f"\n  Charts saved to:  outputs/figures/")
print(f"  Report saved to:  outputs/reports/olist_customer_analysis_report.html")
print("\n  Open the HTML report in any browser to view the full analysis.\n")
