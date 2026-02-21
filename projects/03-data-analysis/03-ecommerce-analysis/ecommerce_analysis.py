#!/usr/bin/env python3
"""
E-Commerce Data Analysis
========================
Generates synthetic e-commerce transaction data and performs comprehensive
analysis including RFM segmentation, cohort analysis, and revenue trends.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_CUSTOMERS = 400
N_ORDERS = 3000
OUTPUT_DIR = "outputs"

np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)


# ===========================================================================
# 1. Synthetic Data Generation
# ===========================================================================
def generate_ecommerce_data(n_customers: int = N_CUSTOMERS,
                            n_orders: int = N_ORDERS) -> pd.DataFrame:
    """Create a synthetic e-commerce transactions dataset."""

    # Date range: 2 years of data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range_days = (end_date - start_date).days

    categories = ["Electronics", "Clothing", "Home & Garden", "Books",
                  "Sports", "Beauty", "Toys", "Food & Beverage"]
    category_weights = np.array([20, 22, 12, 10, 10, 10, 8, 8], dtype=float)
    category_weights /= category_weights.sum()

    products = {
        "Electronics": ["Laptop", "Phone", "Headphones", "Tablet", "Smartwatch",
                        "Camera", "Speaker", "Charger"],
        "Clothing": ["T-Shirt", "Jeans", "Dress", "Jacket", "Sneakers",
                     "Hat", "Scarf", "Socks"],
        "Home & Garden": ["Lamp", "Pillow", "Plant Pot", "Rug", "Candle",
                          "Vase", "Blanket", "Shelf"],
        "Books": ["Fiction Novel", "Cookbook", "Biography", "Sci-Fi Novel",
                  "Textbook", "Art Book", "Travel Guide", "Self-Help"],
        "Sports": ["Yoga Mat", "Dumbbells", "Running Shoes", "Water Bottle",
                   "Resistance Band", "Jump Rope", "Backpack", "Jersey"],
        "Beauty": ["Moisturizer", "Lipstick", "Perfume", "Shampoo",
                   "Sunscreen", "Face Mask", "Nail Polish", "Serum"],
        "Toys": ["Board Game", "Puzzle", "Action Figure", "Lego Set",
                 "Stuffed Animal", "Drone", "Card Game", "Building Blocks"],
        "Food & Beverage": ["Coffee Beans", "Tea Set", "Chocolate Box",
                            "Olive Oil", "Spice Kit", "Wine", "Honey", "Granola"],
    }

    price_ranges = {
        "Electronics": (25, 1200),
        "Clothing": (10, 200),
        "Home & Garden": (8, 150),
        "Books": (5, 60),
        "Sports": (10, 180),
        "Beauty": (5, 120),
        "Toys": (8, 100),
        "Food & Beverage": (5, 80),
    }

    # Assign customers
    customer_ids = np.random.randint(1, n_customers + 1, size=n_orders)
    # Assign first-purchase month for cohort analysis
    customer_first_month = {}
    for cid in range(1, n_customers + 1):
        customer_first_month[cid] = start_date + timedelta(
            days=np.random.randint(0, date_range_days // 2))

    regions = ["North", "South", "East", "West"]
    customer_region = {cid: np.random.choice(regions) for cid in range(1, n_customers + 1)}

    rows = []
    for i in range(n_orders):
        cid = customer_ids[i]
        first = customer_first_month[cid]
        # Order date: between first purchase and end date
        days_avail = (end_date - first).days
        order_date = first + timedelta(days=np.random.randint(0, max(days_avail, 1)))

        cat = np.random.choice(categories, p=category_weights)
        product = np.random.choice(products[cat])
        lo, hi = price_ranges[cat]
        price = round(np.random.uniform(lo, hi), 2)
        quantity = np.random.choice([1, 1, 1, 2, 2, 3, 4], p=[0.40, 0.15, 0.10, 0.15, 0.08, 0.07, 0.05])
        discount = np.random.choice([0, 0, 0, 0, 5, 10, 15, 20, 25],
                                    p=[0.45, 0.10, 0.05, 0.05, 0.10, 0.10, 0.05, 0.05, 0.05])

        rows.append({
            "order_id": i + 1,
            "customer_id": cid,
            "order_date": order_date.strftime("%Y-%m-%d"),
            "category": cat,
            "product": product,
            "quantity": quantity,
            "unit_price": price,
            "discount_pct": discount,
            "region": customer_region[cid],
        })

    df = pd.DataFrame(rows)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["total_price"] = (df["unit_price"] * df["quantity"] *
                         (1 - df["discount_pct"] / 100)).round(2)
    df["order_month"] = df["order_date"].dt.to_period("M")

    return df


# ===========================================================================
# 2. Exploratory Data Analysis
# ===========================================================================
def basic_overview(df: pd.DataFrame) -> None:
    """Print basic summary of the dataset."""
    print("=" * 70)
    print("E-COMMERCE DATASET — BASIC OVERVIEW")
    print("=" * 70)
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['order_date'].min().date()} to {df['order_date'].max().date()}")
    print(f"Unique customers: {df['customer_id'].nunique()}")
    print(f"Total revenue: ${df['total_price'].sum():,.2f}")
    print(f"Average order value: ${df['total_price'].mean():,.2f}")
    print(f"\n--- Orders by Category ---")
    print(df["category"].value_counts())
    print()


def revenue_trends(df: pd.DataFrame) -> None:
    """Analyse revenue over time."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Revenue & Sales Trends", fontsize=18, fontweight="bold", y=1.02)

    # 1. Monthly revenue
    monthly_rev = df.groupby("order_month")["total_price"].sum()
    monthly_rev.index = monthly_rev.index.to_timestamp()
    axes[0, 0].plot(monthly_rev.index, monthly_rev.values, marker="o",
                    linewidth=2, color="steelblue")
    axes[0, 0].fill_between(monthly_rev.index, monthly_rev.values,
                            alpha=0.15, color="steelblue")
    axes[0, 0].set_title("Monthly Revenue", fontsize=14)
    axes[0, 0].set_ylabel("Revenue ($)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. Monthly order count
    monthly_count = df.groupby("order_month")["order_id"].count()
    monthly_count.index = monthly_count.index.to_timestamp()
    axes[0, 1].bar(monthly_count.index, monthly_count.values, width=20,
                   color="coral", alpha=0.8)
    axes[0, 1].set_title("Monthly Order Count", fontsize=14)
    axes[0, 1].set_ylabel("Orders")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # 3. Revenue by category
    cat_rev = df.groupby("category")["total_price"].sum().sort_values(ascending=True)
    cat_rev.plot(kind="barh", color=sns.color_palette("viridis", len(cat_rev)), ax=axes[1, 0])
    axes[1, 0].set_title("Total Revenue by Category", fontsize=14)
    axes[1, 0].set_xlabel("Revenue ($)")

    # 4. Average order value by category
    cat_aov = df.groupby("category")["total_price"].mean().sort_values(ascending=True)
    cat_aov.plot(kind="barh", color=sns.color_palette("magma", len(cat_aov)), ax=axes[1, 1])
    axes[1, 1].set_title("Average Order Value by Category", fontsize=14)
    axes[1, 1].set_xlabel("AOV ($)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/revenue_trends.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] revenue_trends.png")


def rfm_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Perform RFM (Recency, Frequency, Monetary) customer segmentation."""
    reference_date = df["order_date"].max() + timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency=("order_date", lambda x: (reference_date - x.max()).days),
        frequency=("order_id", "count"),
        monetary=("total_price", "sum"),
    ).reset_index()

    # Score each dimension 1-5
    for col in ["recency", "frequency", "monetary"]:
        ascending = col == "recency"  # lower recency is better
        rfm[f"{col}_score"] = pd.qcut(
            rfm[col].rank(method="first", ascending=ascending),
            q=5, labels=[1, 2, 3, 4, 5],
        ).astype(int)

    rfm["rfm_score"] = rfm["recency_score"] + rfm["frequency_score"] + rfm["monetary_score"]

    # Segment labels
    def segment(row):
        s = row["rfm_score"]
        if s >= 13:
            return "Champions"
        elif s >= 10:
            return "Loyal"
        elif s >= 7:
            return "Potential Loyalists"
        elif s >= 5:
            return "At Risk"
        else:
            return "Lost"

    rfm["segment"] = rfm.apply(segment, axis=1)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("RFM Customer Segmentation", fontsize=18, fontweight="bold", y=1.02)

    # 1. Segment distribution
    seg_counts = rfm["segment"].value_counts()
    colors = sns.color_palette("Set2", len(seg_counts))
    axes[0].pie(seg_counts, labels=seg_counts.index, autopct="%1.1f%%",
                colors=colors, textprops={"fontsize": 10})
    axes[0].set_title("Customer Segments", fontsize=14)

    # 2. Recency vs Frequency coloured by monetary
    scatter = axes[1].scatter(rfm["recency"], rfm["frequency"],
                              c=rfm["monetary"], cmap="YlOrRd",
                              alpha=0.6, edgecolors="grey", linewidth=0.3)
    axes[1].set_xlabel("Recency (days)")
    axes[1].set_ylabel("Frequency (orders)")
    axes[1].set_title("Recency vs Frequency (colour = Monetary)", fontsize=14)
    plt.colorbar(scatter, ax=axes[1], label="Monetary ($)")

    # 3. Monetary by segment
    seg_order = ["Champions", "Loyal", "Potential Loyalists", "At Risk", "Lost"]
    present_segs = [s for s in seg_order if s in rfm["segment"].values]
    sns.boxplot(data=rfm, x="segment", y="monetary", order=present_segs,
                palette="viridis", ax=axes[2])
    axes[2].set_title("Monetary Value by Segment", fontsize=14)
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rfm_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] rfm_analysis.png")

    # Summary stats
    print("\n--- RFM Segment Summary ---")
    summary = rfm.groupby("segment").agg(
        count=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    ).round(1)
    print(summary)
    print()

    return rfm


def cohort_analysis(df: pd.DataFrame) -> None:
    """Monthly cohort retention analysis."""
    df_c = df.copy()
    df_c["order_month_ts"] = df_c["order_date"].dt.to_period("M").dt.to_timestamp()

    # First purchase month per customer
    df_c["cohort_month"] = df_c.groupby("customer_id")["order_month_ts"].transform("min")
    df_c["cohort_month"] = pd.to_datetime(df_c["cohort_month"])

    # Period number (months since cohort)
    df_c["period_number"] = (
        (df_c["order_month_ts"].dt.year - df_c["cohort_month"].dt.year) * 12
        + (df_c["order_month_ts"].dt.month - df_c["cohort_month"].dt.month)
    )

    # Cohort table
    cohort_data = (
        df_c.groupby(["cohort_month", "period_number"])["customer_id"]
        .nunique()
        .reset_index(name="customers")
    )
    cohort_pivot = cohort_data.pivot(index="cohort_month", columns="period_number",
                                     values="customers")

    # Retention rates
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0) * 100

    # Limit to first 12 periods for readability and take cohorts with enough data
    retention = retention.iloc[:, :13]
    retention = retention.dropna(how="all")

    # Pick top 12 cohorts by size
    if len(retention) > 12:
        retention = retention.iloc[:12]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(retention, annot=True, fmt=".0f", cmap="YlGnBu",
                linewidths=0.5, ax=ax, vmin=0, vmax=100)
    ax.set_title("Monthly Cohort Retention (%)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("Cohort Month")
    # Format y-axis labels
    labels = [d.strftime("%Y-%m") for d in retention.index]
    ax.set_yticklabels(labels, rotation=0)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cohort_retention.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] cohort_retention.png")


def product_analysis(df: pd.DataFrame) -> None:
    """Product and category deep-dive."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Product & Category Analysis", fontsize=18, fontweight="bold", y=1.02)

    # 1. Top 15 products by revenue
    prod_rev = df.groupby("product")["total_price"].sum().nlargest(15)
    sns.barplot(x=prod_rev.values, y=prod_rev.index, palette="viridis", ax=axes[0, 0])
    axes[0, 0].set_title("Top 15 Products by Revenue", fontsize=14)
    axes[0, 0].set_xlabel("Revenue ($)")

    # 2. Quantity distribution
    sns.histplot(data=df, x="quantity", bins=10, kde=False, color="coral", ax=axes[0, 1])
    axes[0, 1].set_title("Order Quantity Distribution", fontsize=14)

    # 3. Discount usage by category
    df_disc = df[df["discount_pct"] > 0]
    disc_rate = df.groupby("category").apply(
        lambda x: (x["discount_pct"] > 0).mean()
    ).sort_values(ascending=False)
    sns.barplot(x=disc_rate.values, y=disc_rate.index, palette="Oranges_r", ax=axes[1, 0])
    axes[1, 0].set_title("Discount Usage Rate by Category", fontsize=14)
    axes[1, 0].set_xlabel("Proportion with Discount")

    # 4. Revenue by region
    region_rev = df.groupby("region")["total_price"].sum().sort_values(ascending=True)
    region_rev.plot(kind="barh", color=sns.color_palette("Set2", len(region_rev)), ax=axes[1, 1])
    axes[1, 1].set_title("Revenue by Region", fontsize=14)
    axes[1, 1].set_xlabel("Revenue ($)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/product_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] product_analysis.png")


def print_key_insights(df: pd.DataFrame) -> None:
    """Print key findings from the analysis."""
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print(f"  Total orders            : {len(df):,}")
    print(f"  Unique customers        : {df['customer_id'].nunique()}")
    print(f"  Total revenue           : ${df['total_price'].sum():,.2f}")
    print(f"  Average order value     : ${df['total_price'].mean():,.2f}")
    print(f"  Median order value      : ${df['total_price'].median():,.2f}")
    top_cat = df.groupby("category")["total_price"].sum().idxmax()
    print(f"  Top category by revenue : {top_cat}")
    top_prod = df.groupby("product")["total_price"].sum().idxmax()
    print(f"  Top product by revenue  : {top_prod}")
    orders_per_cust = df.groupby("customer_id")["order_id"].count()
    print(f"  Avg orders per customer : {orders_per_cust.mean():.1f}")
    print(f"  Discount usage rate     : {(df['discount_pct'] > 0).mean():.1%}")
    top_region = df.groupby("region")["total_price"].sum().idxmax()
    print(f"  Top region by revenue   : {top_region}")
    print("=" * 70)


# ===========================================================================
# 3. Main
# ===========================================================================
def main() -> None:
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating synthetic e-commerce dataset ...")
    df = generate_ecommerce_data()
    df_save = df.drop(columns=["order_month"])
    df_save.to_csv(f"{OUTPUT_DIR}/ecommerce_synthetic.csv", index=False)
    print(f"[Saved] ecommerce_synthetic.csv  ({len(df)} rows)\n")

    basic_overview(df)
    revenue_trends(df)
    rfm_analysis(df)
    cohort_analysis(df)
    product_analysis(df)
    print_key_insights(df)

    print("\nAll analyses complete. Charts saved to the 'outputs/' directory.")


if __name__ == "__main__":
    main()
