# 🛒 Olist E-Commerce: Customer Behavior & Cohort Analysis

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=flat&logo=pandas)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat&logo=jupyter)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **full end-to-end data analytics project** examining 96,000+ customer transactions from Olist — Brazil's largest e-commerce marketplace. This project demonstrates RFM segmentation, cohort retention analysis, churn identification, and product performance analytics — all from raw CSV data to a polished, shareable HTML report.

---

## 📊 Project Highlights

| Metric | Value |
|--------|-------|
| 📦 Delivered Orders | 96,477 |
| 👤 Unique Customers | 93,357 |
| 💰 Total Revenue | R$ 15.4M |
| 📅 Date Range | Sep 2016 – Aug 2018 |
| ⭐ Avg Review Score | 4.16 / 5.0 |
| 🚚 Avg Delivery Time | 12.1 days |

---

## 🔍 What's Inside

### Five Analysis Modules

| Notebook | Analysis | Key Finding |
|----------|----------|-------------|
| `01_eda.ipynb` | Exploratory Data Analysis | Revenue grew ~300% in 12 months; SP state = 37% of revenue |
| `02_rfm_analysis.ipynb` | RFM Customer Segmentation | 8 segments identified; At-Risk = largest at 24% of customers |
| `03_cohort_analysis.ipynb` | Cohort Retention | Month-1 retention averages just 5% — major loyalty gap |
| `04_churn_analysis.ipynb` | Churn Analysis | 59.2% churned; R$ 8.9M revenue from churned customers |
| `05_product_category_analysis.ipynb` | Category & Payment Analysis | Health & Beauty leads revenue; delivery time drives reviews |

---

## 🏗️ Project Structure

```
olist-ecommerce-analysis/
│
├── 📁 data/
│   └── raw/                            ← Kaggle CSV files (not committed — see setup)
│
├── 📁 src/                             ← Reusable Python modules
│   ├── __init__.py
│   ├── data_loader.py                  ← Dataset loading, merging, feature engineering
│   ├── rfm.py                          ← RFM scoring & segmentation logic
│   ├── cohort.py                       ← Cohort retention & churn analysis
│   └── utils.py                        ← Plotting helpers, color palette, formatters
│
├── 📁 notebooks/                       ← Jupyter notebooks (one per analysis module)
│   ├── 01_eda.ipynb
│   ├── 02_rfm_analysis.ipynb
│   ├── 03_cohort_analysis.ipynb
│   ├── 04_churn_analysis.ipynb
│   └── 05_product_category_analysis.ipynb
│
├── 📁 outputs/
│   ├── figures/                        ← All generated PNG charts (8 charts)
│   └── reports/
│       └── olist_customer_analysis_report.html  ← 📄 Shareable HTML report
│
├── run_analysis.py                     ← Single-command full pipeline runner
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/olist-ecommerce-analysis.git
cd olist-ecommerce-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the **Brazilian E-Commerce Public Dataset by Olist** from Kaggle:
👉 [https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

Extract all CSV files into `data/raw/`. Your folder should look like:
```
data/raw/
├── olist_customers_dataset.csv
├── olist_geolocation_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_orders_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
└── product_category_name_translation.csv
```

### 4. Run the full pipeline
```bash
python run_analysis.py
```

This will generate all 8 charts in `outputs/figures/` and the full HTML report at `outputs/reports/olist_customer_analysis_report.html`.

### 5. Open the report
```bash
# macOS
open outputs/reports/olist_customer_analysis_report.html

# Windows
start outputs/reports/olist_customer_analysis_report.html

# Linux
xdg-open outputs/reports/olist_customer_analysis_report.html
```

### Or explore the notebooks
```bash
jupyter notebook notebooks/
```

---

## 📐 Methodology

### RFM Segmentation
Each customer is scored on three dimensions using **quintile scoring (1–5)**:

| Dimension | Definition | Direction |
|-----------|-----------|-----------|
| **Recency** | Days since last purchase (snapshot = max date + 1) | Lower days → Score 5 |
| **Frequency** | Total number of orders placed | Higher count → Score 5 |
| **Monetary** | Total lifetime spend (R$) | Higher spend → Score 5 |

Scores are combined into 8 named segments using a rule-based classifier:

| Segment | R | F | Description |
|---------|---|---|-------------|
| Champions | ≥4 | ≥4 | Bought recently, buy often, spend the most |
| Loyal Customers | ≥3 | ≥3 | Regular buyers, good spenders |
| Potential Loyalists | ≥3 | ≤2 | Recent buyers with loyalty potential |
| Recent Customers | ≥4 | ≤2 | Just joined, haven't established habits |
| At-Risk | ≤2 | ≥3 | Used to buy often, haven't returned |
| Can't Lose Them | ≤2 | ≤2 | High spend, but going quiet |
| Hibernating | — | — | Low on all dimensions |
| Lost | 1 | 1 | Long gone, lowest scores |

### Cohort Retention
- **Cohort** = customer's first purchase month
- **Cohort index** = months elapsed since first purchase
- **Retention %** = (returning unique customers at index N) / (initial cohort size) × 100

### Churn Definition
A customer is classified as **churned** if their last order was more than **180 days** before the snapshot date. Sensitivity analysis across thresholds from 60–365 days is included in `04_churn_analysis.ipynb`.

---

## 📈 Key Findings & Business Recommendations

### 🔴 Critical: Low Repeat Purchase Rate
Month-1 cohort retention averages only **~5%** — 95% of first-time buyers don't return the following month. A structured post-purchase re-engagement sequence (7/14/30-day email cadence with personalized recommendations) is the highest-priority intervention.

### 🎯 RFM Action Map

| Segment | Action | Priority |
|---------|--------|----------|
| **Champions** | VIP perks, early access, loyalty program | Retention |
| **At-Risk** (22,229 customers) | Win-back campaign with discount + free shipping | 🔴 Urgent |
| **Can't Lose Them** (8,671 customers) | Personal outreach, high-value offers | 🔴 Urgent |
| **Loyal Customers** | Upsell to premium categories, referral program | Growth |
| **Recent Customers** | Second-purchase incentive within 30 days | Conversion |

### 🚚 Delivery → Satisfaction Link
1-star reviews average **20 delivery days** vs. 9 days for 5-star reviews. Reducing delivery variance is the most direct lever to improve NPS, repeat purchases, and public ratings.

### 🗺️ Geographic Expansion
São Paulo + Rio de Janeiro + Minas Gerais = **66% of revenue**. Brazil's North and Northeast regions are structurally underserved — subsidized logistics or regional seller recruitment could open significant new markets.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.11** | Core language |
| **Pandas** | Data loading, merging, aggregation |
| **NumPy** | Numerical operations |
| **Matplotlib + Seaborn** | All chart generation |
| **Scikit-learn** | (Available for churn modeling extensions) |
| **Jupyter Notebook** | Interactive exploration |

---

## 📁 Dataset

**Brazilian E-Commerce Public Dataset by Olist**  
📌 Source: [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
📏 Size: ~120MB across 9 CSV files  
📅 Period: September 2016 – August 2018  
🏢 Publisher: Olist (Brazil's largest department store marketplace)

> The dataset contains ~100,000 orders with product, customer, payment, review, geolocation, and seller information — one of the most comprehensive public e-commerce datasets available.

---

## 🔭 Extensions & Future Work

- [ ] **Churn Prediction Model** — Train a binary classifier (XGBoost / Logistic Regression) on RFM features to predict churn probability per customer
- [ ] **Market Basket Analysis** — Apriori/FP-Growth association rules on order-item pairs for cross-sell recommendations  
- [ ] **Customer LTV Prediction** — BG/NBD or Pareto/NBD model for probabilistic lifetime value estimation
- [ ] **Geospatial Visualization** — Folium/Plotly choropleth maps of revenue density by Brazilian state
- [ ] **Delivery Anomaly Detection** — Flag unusual delivery times for proactive customer communication
- [ ] **Interactive Dashboard** — Convert to Streamlit or Tableau Public for live filtering

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.  
Dataset is publicly available under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license via Kaggle.

---

## 👤 Author

**Amanda**  
MS Data Analytics  
📧 [your.email@example.com] | 🔗 [linkedin.com/in/yourprofile] | 🐙 [github.com/yourusername]

---

*Built as a portfolio project demonstrating end-to-end data analytics: raw data → cleaning → analysis → segmentation → business recommendations → shareable report.*
