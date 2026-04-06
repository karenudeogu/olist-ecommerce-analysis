# Data Directory

## Download Instructions

1. Go to: [https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
2. Download and extract the ZIP file
3. Place all 9 CSV files in this `data/raw/` directory

## Expected Files

```
data/raw/
├── olist_customers_dataset.csv          (~99K rows)
├── olist_geolocation_dataset.csv        (~1M rows)
├── olist_order_items_dataset.csv        (~113K rows)
├── olist_order_payments_dataset.csv     (~104K rows)
├── olist_order_reviews_dataset.csv      (~99K rows)
├── olist_orders_dataset.csv             (~99K rows)
├── olist_products_dataset.csv           (~33K rows)
├── olist_sellers_dataset.csv            (~3K rows)
└── product_category_name_translation.csv (~71 rows)
```

## Dataset Schema

```
orders ──────────┬─── customers
                 ├─── order_items ─── products ─── category_translation
                 ├─── payments
                 └─── reviews

sellers ─────────── order_items
```

Dataset licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) by Olist.
