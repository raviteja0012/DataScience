"""
Data Warehouse Design - Star Schema Implementation
====================================================
Demonstrates data warehousing concepts with:
- Star schema with fact and dimension tables
- SQLite-based implementation
- Synthetic retail data generation
- ETL to populate warehouse from raw sources
- OLAP-style queries (roll-up, drill-down, slice-dice)
- Slowly Changing Dimensions (SCD Type 2)
- Query performance comparison
- Data lineage tracking
"""

import sqlite3
import os
import sys
import json
import csv
import logging
import tempfile
import random
import hashlib
import time
from datetime import datetime, timedelta, date
from collections import defaultdict
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DataWarehouse")


# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------
DIMENSION_SCHEMAS = {
    "dim_product": """
        CREATE TABLE IF NOT EXISTS dim_product (
            product_key INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT NOT NULL,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT,
            brand TEXT,
            unit_price REAL NOT NULL,
            -- SCD Type 2 fields
            effective_date TEXT NOT NULL,
            expiry_date TEXT DEFAULT '9999-12-31',
            is_current INTEGER DEFAULT 1,
            row_hash TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "dim_customer": """
        CREATE TABLE IF NOT EXISTS dim_customer (
            customer_key INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            customer_name TEXT NOT NULL,
            email TEXT,
            segment TEXT,
            city TEXT,
            state TEXT,
            country TEXT DEFAULT 'US',
            -- SCD Type 2 fields
            effective_date TEXT NOT NULL,
            expiry_date TEXT DEFAULT '9999-12-31',
            is_current INTEGER DEFAULT 1,
            row_hash TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "dim_date": """
        CREATE TABLE IF NOT EXISTS dim_date (
            date_key INTEGER PRIMARY KEY,
            full_date TEXT NOT NULL,
            year INTEGER NOT NULL,
            quarter INTEGER NOT NULL,
            month INTEGER NOT NULL,
            month_name TEXT NOT NULL,
            week_of_year INTEGER NOT NULL,
            day_of_month INTEGER NOT NULL,
            day_of_week INTEGER NOT NULL,
            day_name TEXT NOT NULL,
            is_weekend INTEGER NOT NULL,
            is_holiday INTEGER DEFAULT 0,
            fiscal_year INTEGER,
            fiscal_quarter INTEGER
        )
    """,
    "dim_store": """
        CREATE TABLE IF NOT EXISTS dim_store (
            store_key INTEGER PRIMARY KEY AUTOINCREMENT,
            store_id TEXT NOT NULL UNIQUE,
            store_name TEXT NOT NULL,
            store_type TEXT,
            city TEXT,
            state TEXT,
            region TEXT,
            manager TEXT,
            open_date TEXT,
            square_footage INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """,
}

FACT_SCHEMA = """
    CREATE TABLE IF NOT EXISTS fact_sales (
        sale_key INTEGER PRIMARY KEY AUTOINCREMENT,
        date_key INTEGER NOT NULL,
        product_key INTEGER NOT NULL,
        customer_key INTEGER NOT NULL,
        store_key INTEGER NOT NULL,
        order_id TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL,
        discount_amount REAL DEFAULT 0.0,
        sales_amount REAL NOT NULL,
        cost_amount REAL NOT NULL,
        profit_amount REAL NOT NULL,
        FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
        FOREIGN KEY (product_key) REFERENCES dim_product(product_key),
        FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key),
        FOREIGN KEY (store_key) REFERENCES dim_store(store_key)
    )
"""

LINEAGE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS data_lineage (
        lineage_id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_name TEXT NOT NULL,
        operation TEXT NOT NULL,
        source_system TEXT,
        rows_affected INTEGER,
        run_timestamp TEXT NOT NULL,
        run_id TEXT NOT NULL,
        details TEXT
    )
"""


# ---------------------------------------------------------------------------
# Synthetic Data Generator
# ---------------------------------------------------------------------------
class RetailDataGenerator:
    """Generates synthetic retail data for the warehouse."""

    CATEGORIES = {
        "Electronics": ["Laptops", "Phones", "Tablets", "Accessories"],
        "Clothing": ["Shirts", "Pants", "Shoes", "Outerwear"],
        "Home": ["Kitchen", "Furniture", "Decor", "Bedding"],
        "Sports": ["Equipment", "Apparel", "Footwear", "Nutrition"],
    }

    BRANDS = ["Alpha", "BetaTech", "Gamma", "DeltaWear", "Epsilon",
              "ZetaHome", "EtaSport", "ThetaGear"]

    FIRST_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
                   "Grace", "Hank", "Iris", "Jack", "Karen", "Leo",
                   "Mona", "Nate", "Olivia", "Paul"]

    LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones",
                  "Garcia", "Miller", "Davis", "Wilson", "Anderson"]

    SEGMENTS = ["Premium", "Regular", "Budget", "New"]

    CITIES = [
        ("New York", "NY", "Northeast"), ("Los Angeles", "CA", "West"),
        ("Chicago", "IL", "Midwest"), ("Houston", "TX", "South"),
        ("Phoenix", "AZ", "West"), ("Philadelphia", "PA", "Northeast"),
        ("San Antonio", "TX", "South"), ("Dallas", "TX", "South"),
        ("San Jose", "CA", "West"), ("Austin", "TX", "South"),
    ]

    STORE_TYPES = ["Flagship", "Standard", "Outlet", "Express"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_products(self, n: int = 30) -> list[dict]:
        products = []
        pid = 1
        for category, subcats in self.CATEGORIES.items():
            for subcat in subcats:
                for _ in range(n // (len(self.CATEGORIES) * 4) + 1):
                    if len(products) >= n:
                        break
                    products.append({
                        "product_id": f"PROD-{pid:04d}",
                        "product_name": f"{self.rng.choice(self.BRANDS)} {subcat} {pid}",
                        "category": category,
                        "subcategory": subcat,
                        "brand": self.rng.choice(self.BRANDS),
                        "unit_price": round(self.rng.uniform(9.99, 499.99), 2),
                        "cost_ratio": round(self.rng.uniform(0.3, 0.7), 2),
                    })
                    pid += 1
        return products[:n]

    def generate_customers(self, n: int = 50) -> list[dict]:
        customers = []
        for i in range(n):
            first = self.rng.choice(self.FIRST_NAMES)
            last = self.rng.choice(self.LAST_NAMES)
            city, state, region = self.rng.choice(self.CITIES)
            customers.append({
                "customer_id": f"CUST-{i+1:05d}",
                "customer_name": f"{first} {last}",
                "email": f"{first.lower()}.{last.lower()}{i}@email.com",
                "segment": self.rng.choice(self.SEGMENTS),
                "city": city,
                "state": state,
                "region": region,
            })
        return customers

    def generate_stores(self, n: int = 8) -> list[dict]:
        stores = []
        managers = [f"{self.rng.choice(self.FIRST_NAMES)} {self.rng.choice(self.LAST_NAMES)}"
                    for _ in range(n)]
        for i in range(n):
            city, state, region = self.CITIES[i % len(self.CITIES)]
            stores.append({
                "store_id": f"STORE-{i+1:03d}",
                "store_name": f"{city} {self.rng.choice(self.STORE_TYPES)}",
                "store_type": self.rng.choice(self.STORE_TYPES),
                "city": city,
                "state": state,
                "region": region,
                "manager": managers[i],
                "open_date": (
                    datetime(2020, 1, 1) + timedelta(days=self.rng.randint(0, 365))
                ).strftime("%Y-%m-%d"),
                "square_footage": self.rng.randint(2000, 20000),
            })
        return stores

    def generate_sales(self, products: list[dict], customers: list[dict],
                       stores: list[dict], n: int = 500) -> list[dict]:
        sales = []
        for i in range(n):
            product = self.rng.choice(products)
            customer = self.rng.choice(customers)
            store = self.rng.choice(stores)
            qty = self.rng.randint(1, 10)
            price = product["unit_price"]
            discount = round(price * self.rng.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2]), 2)
            sales_amt = round((price - discount) * qty, 2)
            cost_amt = round(price * product["cost_ratio"] * qty, 2)
            order_date = datetime(2024, 1, 1) + timedelta(days=self.rng.randint(0, 364))
            sales.append({
                "order_id": f"ORD-{i+1:06d}",
                "order_date": order_date.strftime("%Y-%m-%d"),
                "product_id": product["product_id"],
                "customer_id": customer["customer_id"],
                "store_id": store["store_id"],
                "quantity": qty,
                "unit_price": price,
                "discount_amount": discount * qty,
                "sales_amount": sales_amt,
                "cost_amount": cost_amt,
                "profit_amount": round(sales_amt - cost_amt, 2),
            })
        return sales


# ---------------------------------------------------------------------------
# Data Warehouse Manager
# ---------------------------------------------------------------------------
class DataWarehouse:
    """Manages the star schema data warehouse."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(
            tempfile.mkdtemp(prefix="warehouse_"), "retail_warehouse.db"
        )
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conn: Optional[sqlite3.Connection] = None
        logger.info(f"Data warehouse: {self.db_path}")

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def _log_lineage(self, table_name: str, operation: str,
                     source: str, rows: int, details: str = ""):
        self.conn.execute(
            "INSERT INTO data_lineage (table_name, operation, source_system, "
            "rows_affected, run_timestamp, run_id, details) VALUES (?,?,?,?,?,?,?)",
            (table_name, operation, source, rows,
             datetime.now().isoformat(), self.run_id, details),
        )

    def create_schema(self):
        """Create all warehouse tables."""
        logger.info("Creating warehouse schema...")
        for name, ddl in DIMENSION_SCHEMAS.items():
            self.conn.execute(ddl)
            logger.info(f"  Created dimension: {name}")
        self.conn.execute(FACT_SCHEMA)
        logger.info("  Created fact table: fact_sales")
        self.conn.execute(LINEAGE_SCHEMA)
        logger.info("  Created lineage table")
        self.conn.commit()

    # -- Date dimension (special: pre-populated) ----------------------------
    def populate_date_dimension(self, start_year: int = 2023, end_year: int = 2025):
        """Populate dim_date with all dates in the range."""
        logger.info(f"Populating dim_date ({start_year}-{end_year})...")
        holidays = {
            (1, 1), (7, 4), (12, 25), (11, 28), (12, 31),
        }
        rows = []
        current = date(start_year, 1, 1)
        end = date(end_year, 12, 31)
        while current <= end:
            date_key = int(current.strftime("%Y%m%d"))
            fiscal_year = current.year if current.month >= 7 else current.year - 1
            fiscal_quarter = ((current.month - 7) % 12) // 3 + 1
            rows.append((
                date_key, current.isoformat(), current.year,
                (current.month - 1) // 3 + 1, current.month,
                current.strftime("%B"), current.isocalendar()[1],
                current.day, current.weekday(),
                current.strftime("%A"),
                1 if current.weekday() >= 5 else 0,
                1 if (current.month, current.day) in holidays else 0,
                fiscal_year, fiscal_quarter,
            ))
            current += timedelta(days=1)

        self.conn.executemany(
            "INSERT OR IGNORE INTO dim_date VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        self._log_lineage("dim_date", "INSERT", "generated", len(rows))
        logger.info(f"  Inserted {len(rows)} date records")

    # -- SCD Type 2 dimension loading ---------------------------------------
    @staticmethod
    def _compute_hash(record: dict, fields: list[str]) -> str:
        content = "|".join(str(record.get(f, "")) for f in sorted(fields))
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def load_dimension_scd2(self, table: str, records: list[dict],
                            natural_key: str, tracked_fields: list[str],
                            all_fields: list[str]):
        """Load a dimension table using SCD Type 2 logic."""
        logger.info(f"Loading {table} with SCD Type 2 ({len(records)} records)...")
        today = datetime.now().strftime("%Y-%m-%d")
        inserted = 0
        updated = 0

        for record in records:
            nk = record[natural_key]
            new_hash = self._compute_hash(record, tracked_fields)

            # Check for existing current record
            cursor = self.conn.execute(
                f"SELECT {', '.join(all_fields)}, row_hash "
                f"FROM {table} WHERE {natural_key}=? AND is_current=1",
                (nk,),
            )
            existing = cursor.fetchone()

            if existing is None:
                # New record - insert
                cols = all_fields + ["effective_date", "is_current", "row_hash"]
                vals = [record.get(f) for f in all_fields] + [today, 1, new_hash]
                placeholders = ", ".join("?" for _ in cols)
                self.conn.execute(
                    f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
                    vals,
                )
                inserted += 1
            elif existing["row_hash"] != new_hash:
                # Changed record - expire old, insert new
                self.conn.execute(
                    f"UPDATE {table} SET is_current=0, expiry_date=? "
                    f"WHERE {natural_key}=? AND is_current=1",
                    (today, nk),
                )
                cols = all_fields + ["effective_date", "is_current", "row_hash"]
                vals = [record.get(f) for f in all_fields] + [today, 1, new_hash]
                placeholders = ", ".join("?" for _ in cols)
                self.conn.execute(
                    f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
                    vals,
                )
                updated += 1
            # else: no change, skip

        self.conn.commit()
        self._log_lineage(
            table, "SCD2_LOAD", "raw_source", inserted + updated,
            f"inserted={inserted}, updated={updated}",
        )
        logger.info(f"  {table}: {inserted} inserts, {updated} SCD2 updates")
        return inserted, updated

    # -- Fact table loading -------------------------------------------------
    def load_fact_sales(self, sales: list[dict]):
        """Load fact_sales by resolving dimension keys."""
        logger.info(f"Loading fact_sales ({len(sales)} records)...")
        loaded = 0
        skipped = 0

        for sale in sales:
            # Resolve date key
            date_key = int(sale["order_date"].replace("-", ""))

            # Resolve product key (current version)
            cursor = self.conn.execute(
                "SELECT product_key FROM dim_product "
                "WHERE product_id=? AND is_current=1",
                (sale["product_id"],),
            )
            prod_row = cursor.fetchone()
            if not prod_row:
                skipped += 1
                continue
            product_key = prod_row["product_key"]

            # Resolve customer key (current version)
            cursor = self.conn.execute(
                "SELECT customer_key FROM dim_customer "
                "WHERE customer_id=? AND is_current=1",
                (sale["customer_id"],),
            )
            cust_row = cursor.fetchone()
            if not cust_row:
                skipped += 1
                continue
            customer_key = cust_row["customer_key"]

            # Resolve store key
            cursor = self.conn.execute(
                "SELECT store_key FROM dim_store WHERE store_id=?",
                (sale["store_id"],),
            )
            store_row = cursor.fetchone()
            if not store_row:
                skipped += 1
                continue
            store_key = store_row["store_key"]

            self.conn.execute(
                "INSERT INTO fact_sales (date_key, product_key, customer_key, "
                "store_key, order_id, quantity, unit_price, discount_amount, "
                "sales_amount, cost_amount, profit_amount) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (date_key, product_key, customer_key, store_key,
                 sale["order_id"], sale["quantity"], sale["unit_price"],
                 sale["discount_amount"], sale["sales_amount"],
                 sale["cost_amount"], sale["profit_amount"]),
            )
            loaded += 1

        self.conn.commit()
        self._log_lineage("fact_sales", "INSERT", "raw_source", loaded,
                          f"loaded={loaded}, skipped={skipped}")
        logger.info(f"  fact_sales: {loaded} loaded, {skipped} skipped")


# ---------------------------------------------------------------------------
# OLAP Query Engine
# ---------------------------------------------------------------------------
class OLAPQueryEngine:
    """Executes OLAP-style analytical queries on the warehouse."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.query_timings: list[dict] = []

    def _timed_query(self, name: str, sql: str, params: tuple = ()) -> list[dict]:
        t0 = time.time()
        cursor = self.conn.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        elapsed = (time.time() - t0) * 1000
        self.query_timings.append({"query": name, "time_ms": round(elapsed, 2), "rows": len(rows)})
        return rows

    def rollup_sales_by_category(self) -> list[dict]:
        """Roll-up: Sales aggregated by product category."""
        sql = """
            SELECT
                dp.category,
                COUNT(*) as num_transactions,
                SUM(fs.quantity) as total_units,
                ROUND(SUM(fs.sales_amount), 2) as total_sales,
                ROUND(SUM(fs.profit_amount), 2) as total_profit,
                ROUND(AVG(fs.sales_amount), 2) as avg_sale
            FROM fact_sales fs
            JOIN dim_product dp ON fs.product_key = dp.product_key
            GROUP BY dp.category
            ORDER BY total_sales DESC
        """
        return self._timed_query("rollup_by_category", sql)

    def rollup_sales_by_quarter(self) -> list[dict]:
        """Roll-up: Sales by year and quarter."""
        sql = """
            SELECT
                dd.year, dd.quarter,
                COUNT(*) as num_transactions,
                ROUND(SUM(fs.sales_amount), 2) as total_sales,
                ROUND(SUM(fs.profit_amount), 2) as total_profit
            FROM fact_sales fs
            JOIN dim_date dd ON fs.date_key = dd.date_key
            GROUP BY dd.year, dd.quarter
            ORDER BY dd.year, dd.quarter
        """
        return self._timed_query("rollup_by_quarter", sql)

    def drilldown_category_to_subcategory(self, category: str) -> list[dict]:
        """Drill-down: From category to subcategory detail."""
        sql = """
            SELECT
                dp.category,
                dp.subcategory,
                dp.brand,
                COUNT(*) as num_transactions,
                ROUND(SUM(fs.sales_amount), 2) as total_sales,
                ROUND(SUM(fs.profit_amount), 2) as total_profit
            FROM fact_sales fs
            JOIN dim_product dp ON fs.product_key = dp.product_key
            WHERE dp.category = ?
            GROUP BY dp.category, dp.subcategory, dp.brand
            ORDER BY total_sales DESC
        """
        return self._timed_query("drilldown_subcategory", sql, (category,))

    def slice_by_region(self, region: str) -> list[dict]:
        """Slice: Filter sales to a specific region."""
        sql = """
            SELECT
                ds.region, ds.store_name,
                dp.category,
                COUNT(*) as num_transactions,
                ROUND(SUM(fs.sales_amount), 2) as total_sales
            FROM fact_sales fs
            JOIN dim_store ds ON fs.store_key = ds.store_key
            JOIN dim_product dp ON fs.product_key = dp.product_key
            WHERE ds.region = ?
            GROUP BY ds.store_name, dp.category
            ORDER BY total_sales DESC
        """
        return self._timed_query("slice_by_region", sql, (region,))

    def dice_segment_category(self, segments: list[str],
                              categories: list[str]) -> list[dict]:
        """Dice: Filter on multiple dimensions simultaneously."""
        seg_ph = ", ".join("?" for _ in segments)
        cat_ph = ", ".join("?" for _ in categories)
        sql = f"""
            SELECT
                dc.segment,
                dp.category,
                dd.quarter,
                COUNT(*) as num_transactions,
                ROUND(SUM(fs.sales_amount), 2) as total_sales,
                ROUND(AVG(fs.sales_amount), 2) as avg_sale
            FROM fact_sales fs
            JOIN dim_customer dc ON fs.customer_key = dc.customer_key
            JOIN dim_product dp ON fs.product_key = dp.product_key
            JOIN dim_date dd ON fs.date_key = dd.date_key
            WHERE dc.segment IN ({seg_ph}) AND dp.category IN ({cat_ph})
            GROUP BY dc.segment, dp.category, dd.quarter
            ORDER BY dc.segment, dp.category, dd.quarter
        """
        return self._timed_query("dice_segment_category", sql,
                                 tuple(segments + categories))

    def top_customers(self, limit: int = 10) -> list[dict]:
        """Top N customers by total spend."""
        sql = """
            SELECT
                dc.customer_name,
                dc.segment,
                dc.city,
                COUNT(DISTINCT fs.order_id) as num_orders,
                SUM(fs.quantity) as total_units,
                ROUND(SUM(fs.sales_amount), 2) as total_spend,
                ROUND(AVG(fs.sales_amount), 2) as avg_order_value
            FROM fact_sales fs
            JOIN dim_customer dc ON fs.customer_key = dc.customer_key
            GROUP BY dc.customer_key
            ORDER BY total_spend DESC
            LIMIT ?
        """
        return self._timed_query("top_customers", sql, (limit,))

    def monthly_trend(self) -> list[dict]:
        """Monthly sales trend."""
        sql = """
            SELECT
                dd.year, dd.month, dd.month_name,
                COUNT(*) as num_transactions,
                ROUND(SUM(fs.sales_amount), 2) as total_sales,
                ROUND(SUM(fs.profit_amount), 2) as total_profit,
                ROUND(SUM(fs.profit_amount) * 100.0 / NULLIF(SUM(fs.sales_amount), 0), 1) as margin_pct
            FROM fact_sales fs
            JOIN dim_date dd ON fs.date_key = dd.date_key
            GROUP BY dd.year, dd.month
            ORDER BY dd.year, dd.month
        """
        return self._timed_query("monthly_trend", sql)

    def store_performance(self) -> list[dict]:
        """Store performance comparison."""
        sql = """
            SELECT
                ds.store_name,
                ds.store_type,
                ds.region,
                COUNT(*) as num_transactions,
                ROUND(SUM(fs.sales_amount), 2) as total_sales,
                ROUND(SUM(fs.profit_amount), 2) as total_profit,
                ROUND(SUM(fs.sales_amount) / ds.square_footage, 2) as sales_per_sqft
            FROM fact_sales fs
            JOIN dim_store ds ON fs.store_key = ds.store_key
            GROUP BY ds.store_key
            ORDER BY total_sales DESC
        """
        return self._timed_query("store_performance", sql)

    def performance_report(self) -> str:
        lines = ["\n=== Query Performance Report ==="]
        for qt in self.query_timings:
            lines.append(f"  {qt['query']:30s}: {qt['time_ms']:7.2f} ms ({qt['rows']} rows)")
        total_ms = sum(q["time_ms"] for q in self.query_timings)
        lines.append(f"  {'TOTAL':30s}: {total_ms:7.2f} ms")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
class WarehousePipeline:
    """Full pipeline: generate data, build warehouse, run analytics."""

    def __init__(self, db_path: Optional[str] = None):
        self.warehouse = DataWarehouse(db_path)
        self.generator = RetailDataGenerator(seed=42)

    def run(self):
        """Execute full warehouse build and analytics."""
        logger.info("=" * 60)
        logger.info("DATA WAREHOUSE PIPELINE")
        logger.info("=" * 60)
        t_start = time.time()

        self.warehouse.connect()
        conn = self.warehouse.conn

        try:
            # ---- CREATE SCHEMA ----
            self.warehouse.create_schema()

            # ---- GENERATE RAW DATA ----
            logger.info("\nGenerating synthetic retail data...")
            products = self.generator.generate_products(30)
            customers = self.generator.generate_customers(50)
            stores = self.generator.generate_stores(8)
            sales = self.generator.generate_sales(products, customers, stores, 500)
            logger.info(
                f"  Generated: {len(products)} products, {len(customers)} customers, "
                f"{len(stores)} stores, {len(sales)} sales"
            )

            # ---- LOAD DIMENSIONS ----
            logger.info("\nLoading dimensions...")
            self.warehouse.populate_date_dimension(2023, 2025)

            self.warehouse.load_dimension_scd2(
                "dim_product", products, "product_id",
                ["product_name", "category", "subcategory", "brand", "unit_price"],
                ["product_id", "product_name", "category", "subcategory", "brand", "unit_price"],
            )

            self.warehouse.load_dimension_scd2(
                "dim_customer", customers, "customer_id",
                ["customer_name", "email", "segment", "city", "state"],
                ["customer_id", "customer_name", "email", "segment", "city", "state", "country"],
            )

            # Load stores (simple insert, no SCD)
            for store in stores:
                conn.execute(
                    "INSERT OR IGNORE INTO dim_store "
                    "(store_id, store_name, store_type, city, state, region, "
                    "manager, open_date, square_footage) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (store["store_id"], store["store_name"], store["store_type"],
                     store["city"], store["state"], store["region"],
                     store["manager"], store["open_date"], store["square_footage"]),
                )
            conn.commit()
            self.warehouse._log_lineage("dim_store", "INSERT", "raw_source", len(stores))
            logger.info(f"  dim_store: {len(stores)} loaded")

            # ---- SIMULATE SCD2 UPDATE ----
            logger.info("\nSimulating SCD Type 2 changes...")
            updated_products = []
            for p in products[:3]:
                updated = dict(p)
                updated["unit_price"] = round(p["unit_price"] * 1.1, 2)  # 10% price increase
                updated_products.append(updated)
            ins, upd = self.warehouse.load_dimension_scd2(
                "dim_product", updated_products, "product_id",
                ["product_name", "category", "subcategory", "brand", "unit_price"],
                ["product_id", "product_name", "category", "subcategory", "brand", "unit_price"],
            )
            logger.info(f"  SCD2 demo: {ins} inserts, {upd} version updates")

            # Show SCD2 history
            cursor = conn.execute(
                "SELECT product_id, product_name, unit_price, effective_date, "
                "expiry_date, is_current FROM dim_product "
                "WHERE product_id IN (SELECT product_id FROM dim_product "
                "GROUP BY product_id HAVING COUNT(*) > 1) "
                "ORDER BY product_id, effective_date"
            )
            scd2_history = [dict(row) for row in cursor.fetchall()]
            if scd2_history:
                logger.info("  SCD2 history sample:")
                for h in scd2_history[:6]:
                    logger.info(
                        f"    {h['product_id']}: ${h['unit_price']} "
                        f"[{h['effective_date']} - {h['expiry_date']}] "
                        f"current={h['is_current']}"
                    )

            # ---- LOAD FACTS ----
            self.warehouse.load_fact_sales(sales)

            # ---- CREATE INDEXES ----
            logger.info("\nCreating indexes...")
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_fact_date ON fact_sales(date_key)",
                "CREATE INDEX IF NOT EXISTS idx_fact_product ON fact_sales(product_key)",
                "CREATE INDEX IF NOT EXISTS idx_fact_customer ON fact_sales(customer_key)",
                "CREATE INDEX IF NOT EXISTS idx_fact_store ON fact_sales(store_key)",
                "CREATE INDEX IF NOT EXISTS idx_dim_prod_current ON dim_product(product_id, is_current)",
                "CREATE INDEX IF NOT EXISTS idx_dim_cust_current ON dim_customer(customer_id, is_current)",
            ]
            for idx_sql in indexes:
                conn.execute(idx_sql)
            conn.commit()
            logger.info(f"  Created {len(indexes)} indexes")

            # ---- ANALYTICS ----
            logger.info("\n" + "=" * 60)
            logger.info("OLAP ANALYTICS")
            logger.info("=" * 60)
            olap = OLAPQueryEngine(conn)

            # Roll-up by category
            print("\n--- Roll-Up: Sales by Category ---")
            for row in olap.rollup_sales_by_category():
                print(
                    f"  {row['category']:15s}: ${row['total_sales']:>10,.2f} "
                    f"({row['num_transactions']} txns, "
                    f"profit: ${row['total_profit']:>10,.2f})"
                )

            # Roll-up by quarter
            print("\n--- Roll-Up: Sales by Quarter ---")
            for row in olap.rollup_sales_by_quarter():
                print(
                    f"  {row['year']} Q{row['quarter']}: ${row['total_sales']:>10,.2f} "
                    f"({row['num_transactions']} txns)"
                )

            # Drill-down
            print("\n--- Drill-Down: Electronics by Subcategory ---")
            for row in olap.drilldown_category_to_subcategory("Electronics"):
                print(
                    f"  {row['subcategory']:15s} ({row['brand']:12s}): "
                    f"${row['total_sales']:>10,.2f}"
                )

            # Slice
            print("\n--- Slice: South Region Sales ---")
            for row in olap.slice_by_region("South"):
                print(
                    f"  {row['store_name']:25s} | {row['category']:15s}: "
                    f"${row['total_sales']:>10,.2f}"
                )

            # Dice
            print("\n--- Dice: Premium/Regular x Electronics/Clothing ---")
            for row in olap.dice_segment_category(
                ["Premium", "Regular"], ["Electronics", "Clothing"]
            ):
                print(
                    f"  {row['segment']:10s} | {row['category']:12s} | Q{row['quarter']}: "
                    f"${row['total_sales']:>8,.2f} (avg: ${row['avg_sale']:>6,.2f})"
                )

            # Top customers
            print("\n--- Top 10 Customers ---")
            for i, row in enumerate(olap.top_customers(10), 1):
                print(
                    f"  {i:2d}. {row['customer_name']:20s} ({row['segment']:8s}): "
                    f"${row['total_spend']:>10,.2f} ({row['num_orders']} orders)"
                )

            # Monthly trend
            print("\n--- Monthly Sales Trend ---")
            for row in olap.monthly_trend():
                bar = "#" * int(row["total_sales"] / 1000)
                print(
                    f"  {row['year']}-{row['month']:02d} ({row['month_name'][:3]}): "
                    f"${row['total_sales']:>10,.2f} | {bar}"
                )

            # Store performance
            print("\n--- Store Performance ---")
            for row in olap.store_performance():
                print(
                    f"  {row['store_name']:25s} ({row['store_type']:10s}): "
                    f"${row['total_sales']:>10,.2f}  "
                    f"${row['sales_per_sqft']}/sqft"
                )

            # Performance report
            print(olap.performance_report())

            # ---- DATA LINEAGE ----
            print("\n--- Data Lineage ---")
            cursor = conn.execute(
                "SELECT * FROM data_lineage ORDER BY lineage_id"
            )
            for row in cursor.fetchall():
                row = dict(row)
                print(
                    f"  [{row['operation']:12s}] {row['table_name']:15s}: "
                    f"{row['rows_affected']} rows from {row['source_system']} "
                    f"({row['details']})"
                )

            # ---- WAREHOUSE STATS ----
            total_time = time.time() - t_start
            print("\n" + "=" * 70)
            print("WAREHOUSE SUMMARY")
            print("=" * 70)
            tables = ["dim_date", "dim_product", "dim_customer", "dim_store", "fact_sales"]
            for tbl in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                print(f"  {tbl:20s}: {count:>6d} rows")
            print(f"\n  Database file : {self.warehouse.db_path}")
            db_size = os.path.getsize(self.warehouse.db_path)
            print(f"  Database size : {db_size / 1024:.1f} KB")
            print(f"  Total time    : {total_time:.2f} seconds")
            print("=" * 70)
            print("Data warehouse pipeline completed successfully!")
            print("=" * 70)

        finally:
            self.warehouse.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Data Warehouse Design - Data Engineering Portfolio Project")
    print("-" * 55)
    pipeline = WarehousePipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
