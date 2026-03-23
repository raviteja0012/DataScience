"""
ETL Pipeline - Complete Extract, Transform, Load Framework
===========================================================
Demonstrates a production-style ETL pipeline with:
- Multi-source extraction (CSV, JSON, simulated API)
- Comprehensive data transformation (cleaning, dedup, validation, enrichment)
- Multi-target loading (CSV, SQLite)
- Error handling, logging, and data quality checks
- Pipeline configuration and execution reporting
"""

import csv
import json
import sqlite3
import logging
import os
import sys
import tempfile
import hashlib
import re
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Any, Optional
import random
import io

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ETLPipeline")

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "pipeline_name": "customer_orders_etl",
    "version": "1.0.0",
    "sources": {
        "customers_csv": {"type": "csv", "path": "customers.csv"},
        "orders_json": {"type": "json", "path": "orders.json"},
        "products_api": {"type": "api", "endpoint": "/api/v1/products"},
    },
    "targets": {
        "sqlite": {"type": "sqlite", "database": "warehouse.db"},
        "csv_export": {"type": "csv", "directory": "exports"},
    },
    "transform": {
        "dedup_keys": {"customers": ["email"], "orders": ["order_id"]},
        "required_fields": {
            "customers": ["customer_id", "name", "email"],
            "orders": ["order_id", "customer_id", "product_id", "quantity"],
            "products": ["product_id", "name", "price"],
        },
        "type_casts": {
            "customers": {"customer_id": "int", "age": "int"},
            "orders": {
                "order_id": "int",
                "customer_id": "int",
                "product_id": "int",
                "quantity": "int",
                "total_amount": "float",
            },
            "products": {"product_id": "int", "price": "float", "stock": "int"},
        },
    },
    "quality": {
        "min_row_count": {"customers": 5, "orders": 5, "products": 3},
        "max_null_pct": 0.1,
        "unique_keys": {"customers": ["email"], "orders": ["order_id"]},
    },
}


# ---------------------------------------------------------------------------
# Synthetic data generators (stand-in for real sources)
# ---------------------------------------------------------------------------
class SyntheticDataGenerator:
    """Generates realistic synthetic data for demonstration purposes."""

    FIRST_NAMES = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace",
        "Hank", "Iris", "Jack", "Karen", "Leo", "Mona", "Nate",
        "Olivia", "Paul", "Quinn", "Rose", "Steve", "Tina",
    ]
    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
        "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez",
        "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
    ]
    DOMAINS = ["email.com", "mail.org", "inbox.net", "postbox.io"]
    PRODUCT_NAMES = [
        "Wireless Mouse", "Mechanical Keyboard", "USB-C Hub", "Monitor Stand",
        "Webcam HD", "Noise-Cancelling Headphones", "Laptop Sleeve",
        "Desk Lamp", "Ergonomic Chair Pad", "Cable Management Kit",
    ]
    CATEGORIES = ["Electronics", "Accessories", "Furniture", "Office Supplies"]
    CITIES = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin",
    ]
    STATES = ["NY", "CA", "IL", "TX", "AZ", "PA", "TX", "CA", "TX", "TX"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    # -- customers ----------------------------------------------------------
    def generate_customers(self, n: int = 20) -> list[dict]:
        customers: list[dict] = []
        used_emails: set[str] = set()
        cid = 1000
        for _ in range(n):
            first = self.rng.choice(self.FIRST_NAMES)
            last = self.rng.choice(self.LAST_NAMES)
            email = f"{first.lower()}.{last.lower()}@{self.rng.choice(self.DOMAINS)}"
            # Intentionally introduce some duplicates
            if self.rng.random() < 0.15 and used_emails:
                email = self.rng.choice(list(used_emails))
            used_emails.add(email)
            idx = self.rng.randint(0, len(self.CITIES) - 1)
            age = self.rng.randint(18, 75)
            # Introduce occasional bad data
            if self.rng.random() < 0.08:
                age = None  # missing age
            if self.rng.random() < 0.05:
                email = "INVALID"  # bad email
            customers.append({
                "customer_id": cid,
                "name": f"{first} {last}",
                "email": email,
                "age": age,
                "city": self.CITIES[idx],
                "state": self.STATES[idx],
                "signup_date": (
                    datetime(2023, 1, 1)
                    + timedelta(days=self.rng.randint(0, 730))
                ).strftime("%Y-%m-%d"),
            })
            cid += 1
        return customers

    # -- orders -------------------------------------------------------------
    def generate_orders(self, customers: list[dict], products: list[dict],
                        n: int = 50) -> list[dict]:
        orders: list[dict] = []
        oid = 5000
        cust_ids = [c["customer_id"] for c in customers]
        prod_map = {p["product_id"]: p["price"] for p in products}
        prod_ids = list(prod_map.keys())
        for _ in range(n):
            pid = self.rng.choice(prod_ids)
            qty = self.rng.randint(1, 10)
            price = prod_map[pid]
            total = round(price * qty, 2)
            # Occasional bad data
            if self.rng.random() < 0.05:
                total = -abs(total)  # negative total (error)
            order = {
                "order_id": oid,
                "customer_id": self.rng.choice(cust_ids),
                "product_id": pid,
                "quantity": qty,
                "total_amount": total,
                "order_date": (
                    datetime(2024, 1, 1)
                    + timedelta(days=self.rng.randint(0, 365))
                ).strftime("%Y-%m-%d"),
                "status": self.rng.choice(
                    ["completed", "completed", "completed", "pending",
                     "shipped", "cancelled", "returned"]
                ),
            }
            orders.append(order)
            oid += 1
            # Introduce duplicate
            if self.rng.random() < 0.08:
                orders.append(dict(order))
        return orders

    # -- products -----------------------------------------------------------
    def generate_products(self, n: int = 10) -> list[dict]:
        products: list[dict] = []
        pid = 100
        for i in range(min(n, len(self.PRODUCT_NAMES))):
            products.append({
                "product_id": pid,
                "name": self.PRODUCT_NAMES[i],
                "category": self.rng.choice(self.CATEGORIES),
                "price": round(self.rng.uniform(9.99, 299.99), 2),
                "stock": self.rng.randint(0, 500),
            })
            pid += 1
        return products


# ---------------------------------------------------------------------------
# Data Quality Checker
# ---------------------------------------------------------------------------
class DataQualityChecker:
    """Runs quality checks on datasets and collects results."""

    def __init__(self, config: dict):
        self.config = config.get("quality", {})
        self.results: list[dict] = []

    def check_row_count(self, dataset_name: str, data: list[dict]) -> bool:
        min_rows = self.config.get("min_row_count", {}).get(dataset_name, 0)
        passed = len(data) >= min_rows
        self.results.append({
            "check": "row_count",
            "dataset": dataset_name,
            "expected": f">= {min_rows}",
            "actual": len(data),
            "passed": passed,
        })
        return passed

    def check_null_percentage(self, dataset_name: str, data: list[dict]) -> bool:
        if not data:
            return True
        max_pct = self.config.get("max_null_pct", 0.1)
        total_cells = len(data) * len(data[0])
        null_cells = sum(
            1 for row in data for v in row.values() if v is None or v == ""
        )
        pct = null_cells / total_cells if total_cells else 0
        passed = pct <= max_pct
        self.results.append({
            "check": "null_percentage",
            "dataset": dataset_name,
            "expected": f"<= {max_pct:.0%}",
            "actual": f"{pct:.2%}",
            "passed": passed,
        })
        return passed

    def check_uniqueness(self, dataset_name: str, data: list[dict]) -> bool:
        keys = self.config.get("unique_keys", {}).get(dataset_name, [])
        if not keys or not data:
            return True
        seen: set[str] = set()
        dups = 0
        for row in data:
            key_val = "|".join(str(row.get(k, "")) for k in keys)
            if key_val in seen:
                dups += 1
            seen.add(key_val)
        passed = dups == 0
        self.results.append({
            "check": "uniqueness",
            "dataset": dataset_name,
            "keys": keys,
            "duplicates_found": dups,
            "passed": passed,
        })
        return passed

    def check_required_fields(self, dataset_name: str, data: list[dict],
                              required: list[str]) -> bool:
        if not data:
            return True
        missing = [f for f in required if f not in data[0]]
        passed = len(missing) == 0
        self.results.append({
            "check": "required_fields",
            "dataset": dataset_name,
            "missing_fields": missing,
            "passed": passed,
        })
        return passed

    def run_all_checks(self, dataset_name: str, data: list[dict],
                       required_fields: Optional[list[str]] = None) -> bool:
        r1 = self.check_row_count(dataset_name, data)
        r2 = self.check_null_percentage(dataset_name, data)
        r3 = self.check_uniqueness(dataset_name, data)
        r4 = True
        if required_fields:
            r4 = self.check_required_fields(dataset_name, data, required_fields)
        return all([r1, r2, r3, r4])

    def summary(self) -> str:
        lines = ["\n=== Data Quality Report ==="]
        for r in self.results:
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(f"  [{status}] {r['dataset']}.{r['check']} => {r}")
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        lines.append(f"\n  Total: {passed}/{total} checks passed")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extract Stage
# ---------------------------------------------------------------------------
class Extractor:
    """Extracts data from multiple source types."""

    def __init__(self, config: dict, working_dir: str):
        self.config = config
        self.working_dir = working_dir
        self.stats: dict[str, Any] = {}

    def extract_csv(self, path: str) -> list[dict]:
        full_path = os.path.join(self.working_dir, path)
        logger.info(f"Extracting CSV from {full_path}")
        with open(full_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        self.stats[path] = {"rows": len(data), "source_type": "csv"}
        logger.info(f"  Extracted {len(data)} rows from CSV")
        return data

    def extract_json(self, path: str) -> list[dict]:
        full_path = os.path.join(self.working_dir, path)
        logger.info(f"Extracting JSON from {full_path}")
        with open(full_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("records", data.get("data", [data]))
        self.stats[path] = {"rows": len(data), "source_type": "json"}
        logger.info(f"  Extracted {len(data)} rows from JSON")
        return data

    def extract_api(self, endpoint: str, mock_data: list[dict]) -> list[dict]:
        """Simulate an API extraction using mock data."""
        logger.info(f"Extracting from API endpoint {endpoint}")
        # In production, this would use requests.get(endpoint)
        data = mock_data
        self.stats[endpoint] = {"rows": len(data), "source_type": "api"}
        logger.info(f"  Extracted {len(data)} rows from API")
        return data

    def extract_all(self, mock_api_data: Optional[dict] = None) -> dict[str, list[dict]]:
        mock_api_data = mock_api_data or {}
        datasets: dict[str, list[dict]] = {}
        sources = self.config.get("sources", {})
        for name, src in sources.items():
            try:
                if src["type"] == "csv":
                    datasets[name] = self.extract_csv(src["path"])
                elif src["type"] == "json":
                    datasets[name] = self.extract_json(src["path"])
                elif src["type"] == "api":
                    datasets[name] = self.extract_api(
                        src["endpoint"], mock_api_data.get(name, [])
                    )
                else:
                    logger.warning(f"Unknown source type: {src['type']}")
            except Exception as e:
                logger.error(f"Extraction failed for {name}: {e}")
                datasets[name] = []
        return datasets


# ---------------------------------------------------------------------------
# Transform Stage
# ---------------------------------------------------------------------------
class Transformer:
    """Applies cleaning, deduplication, type-casting, validation, enrichment."""

    def __init__(self, config: dict):
        self.config = config.get("transform", {})
        self.stats: dict[str, dict] = {}

    def _safe_cast(self, value: Any, target_type: str) -> Any:
        if value is None or value == "":
            return None
        try:
            if target_type == "int":
                return int(float(value))
            elif target_type == "float":
                return float(value)
            elif target_type == "str":
                return str(value)
            elif target_type == "bool":
                return str(value).lower() in ("true", "1", "yes")
        except (ValueError, TypeError):
            return None
        return value

    def deduplicate(self, data: list[dict], keys: list[str]) -> list[dict]:
        seen: set[str] = set()
        deduped: list[dict] = []
        dups = 0
        for row in data:
            key_val = "|".join(str(row.get(k, "")) for k in keys)
            if key_val not in seen:
                seen.add(key_val)
                deduped.append(row)
            else:
                dups += 1
        if dups:
            logger.info(f"  Removed {dups} duplicate rows (keys: {keys})")
        return deduped

    def cast_types(self, data: list[dict], type_map: dict[str, str]) -> list[dict]:
        cast_errors = 0
        for row in data:
            for col, ttype in type_map.items():
                if col in row:
                    original = row[col]
                    row[col] = self._safe_cast(row[col], ttype)
                    if row[col] is None and original is not None and original != "":
                        cast_errors += 1
        if cast_errors:
            logger.warning(f"  {cast_errors} type-cast errors (set to None)")
        return data

    def validate_required(self, data: list[dict],
                          required: list[str]) -> tuple[list[dict], list[dict]]:
        valid: list[dict] = []
        rejected: list[dict] = []
        for row in data:
            missing = [f for f in required if row.get(f) is None or row.get(f) == ""]
            if missing:
                row["_rejection_reason"] = f"missing fields: {missing}"
                rejected.append(row)
            else:
                valid.append(row)
        if rejected:
            logger.info(f"  Rejected {len(rejected)} rows with missing required fields")
        return valid, rejected

    def clean_strings(self, data: list[dict]) -> list[dict]:
        for row in data:
            for k, v in row.items():
                if isinstance(v, str):
                    row[k] = v.strip()
        return data

    def validate_emails(self, data: list[dict], email_field: str = "email") -> list[dict]:
        email_re = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
        fixed = 0
        for row in data:
            if email_field in row and row[email_field]:
                if not email_re.match(str(row[email_field])):
                    row[email_field] = None
                    fixed += 1
        if fixed:
            logger.info(f"  Nullified {fixed} invalid email addresses")
        return data

    def enrich_orders(self, orders: list[dict], products: list[dict]) -> list[dict]:
        """Enrich orders with product name and category."""
        prod_lookup = {p["product_id"]: p for p in products}
        for order in orders:
            pid = order.get("product_id")
            prod = prod_lookup.get(pid, {})
            order["product_name"] = prod.get("name", "Unknown")
            order["product_category"] = prod.get("category", "Unknown")
        logger.info("  Enriched orders with product details")
        return orders

    def fix_negative_amounts(self, data: list[dict], field: str = "total_amount") -> list[dict]:
        fixed = 0
        for row in data:
            if field in row and row[field] is not None and row[field] < 0:
                row[field] = abs(row[field])
                row["_amount_corrected"] = True
                fixed += 1
        if fixed:
            logger.info(f"  Corrected {fixed} negative amounts")
        return data

    def add_row_hash(self, data: list[dict]) -> list[dict]:
        """Add a deterministic hash for each row for change detection."""
        for row in data:
            content = "|".join(
                str(row.get(k, ""))
                for k in sorted(row.keys())
                if not k.startswith("_")
            )
            row["_row_hash"] = hashlib.md5(content.encode()).hexdigest()[:12]
        return data

    def transform_dataset(self, name: str, data: list[dict],
                          products: Optional[list[dict]] = None) -> tuple[list[dict], list[dict]]:
        logger.info(f"Transforming dataset: {name} ({len(data)} rows)")
        initial_count = len(data)
        rejected: list[dict] = []

        # 1. Clean strings
        data = self.clean_strings(data)

        # 2. Deduplication
        dedup_keys = self.config.get("dedup_keys", {}).get(name, [])
        if dedup_keys:
            data = self.deduplicate(data, dedup_keys)

        # 3. Type casting
        type_map = self.config.get("type_casts", {}).get(name, {})
        if type_map:
            data = self.cast_types(data, type_map)

        # 4. Required field validation
        required = self.config.get("required_fields", {}).get(name, [])
        if required:
            data, rejected = self.validate_required(data, required)

        # 5. Dataset-specific transforms
        if name == "customers":
            data = self.validate_emails(data)
        elif name == "orders":
            data = self.fix_negative_amounts(data)
            if products:
                data = self.enrich_orders(data, products)

        # 6. Add row hash
        data = self.add_row_hash(data)

        self.stats[name] = {
            "input_rows": initial_count,
            "output_rows": len(data),
            "rejected_rows": len(rejected),
            "dedup_removed": initial_count - len(data) - len(rejected),
        }
        logger.info(
            f"  Transform complete: {initial_count} -> {len(data)} rows "
            f"({len(rejected)} rejected)"
        )
        return data, rejected


# ---------------------------------------------------------------------------
# Load Stage
# ---------------------------------------------------------------------------
class Loader:
    """Loads data to CSV files and SQLite database."""

    def __init__(self, config: dict, working_dir: str):
        self.config = config
        self.working_dir = working_dir
        self.stats: dict[str, dict] = {}

    def load_to_csv(self, data: list[dict], filename: str) -> str:
        export_dir = os.path.join(
            self.working_dir,
            self.config.get("targets", {}).get("csv_export", {}).get("directory", "exports"),
        )
        os.makedirs(export_dir, exist_ok=True)
        filepath = os.path.join(export_dir, filename)
        if not data:
            logger.warning(f"No data to write to {filepath}")
            return filepath
        # Filter out internal fields
        fields = [k for k in data[0].keys() if not k.startswith("_")]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
        self.stats[filename] = {"rows": len(data), "target": "csv", "path": filepath}
        logger.info(f"  Loaded {len(data)} rows to CSV: {filepath}")
        return filepath

    def load_to_sqlite(self, data: list[dict], table_name: str) -> None:
        db_path = os.path.join(
            self.working_dir,
            self.config.get("targets", {}).get("sqlite", {}).get("database", "warehouse.db"),
        )
        if not data:
            logger.warning(f"No data to load into table {table_name}")
            return
        # Filter out internal fields
        fields = [k for k in data[0].keys() if not k.startswith("_")]
        conn = sqlite3.connect(db_path)
        try:
            # Create table
            col_defs = ", ".join(f'"{f}" TEXT' for f in fields)
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} ({col_defs})")
            # Insert rows
            placeholders = ", ".join("?" for _ in fields)
            insert_sql = f"INSERT INTO {table_name} ({', '.join(fields)}) VALUES ({placeholders})"
            rows_to_insert = [
                tuple(row.get(f) for f in fields) for row in data
            ]
            conn.executemany(insert_sql, rows_to_insert)
            conn.commit()
            self.stats[table_name] = {
                "rows": len(data),
                "target": "sqlite",
                "path": db_path,
            }
            logger.info(f"  Loaded {len(data)} rows to SQLite table: {table_name}")
        finally:
            conn.close()

    def load_all(self, datasets: dict[str, list[dict]]) -> None:
        for name, data in datasets.items():
            self.load_to_csv(data, f"{name}_clean.csv")
            self.load_to_sqlite(data, name)


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
class ETLPipeline:
    """Orchestrates the full ETL pipeline with reporting."""

    def __init__(self, config: Optional[dict] = None, working_dir: Optional[str] = None):
        self.config = config or DEFAULT_CONFIG
        self.working_dir = working_dir or tempfile.mkdtemp(prefix="etl_")
        self.extractor = Extractor(self.config, self.working_dir)
        self.transformer = Transformer(self.config)
        self.loader = Loader(self.config, self.working_dir)
        self.quality_checker = DataQualityChecker(self.config)
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.stage_times: dict[str, float] = {}
        self.rejected: dict[str, list[dict]] = {}

    def _prepare_synthetic_sources(self) -> dict:
        """Generate synthetic data and write source files."""
        gen = SyntheticDataGenerator(seed=42)
        products = gen.generate_products(10)
        customers = gen.generate_customers(20)
        orders = gen.generate_orders(customers, products, 50)

        # Write customers CSV
        csv_path = os.path.join(self.working_dir, "customers.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=customers[0].keys())
            writer.writeheader()
            writer.writerows(customers)

        # Write orders JSON
        json_path = os.path.join(self.working_dir, "orders.json")
        with open(json_path, "w") as f:
            json.dump({"records": orders}, f, indent=2)

        logger.info("Prepared synthetic source files")
        return {"products_api": products}

    def run(self) -> dict:
        """Execute the full ETL pipeline."""
        self.start_time = datetime.now()
        report: dict[str, Any] = {
            "pipeline": self.config["pipeline_name"],
            "version": self.config["version"],
            "start_time": self.start_time.isoformat(),
            "working_dir": self.working_dir,
            "stages": {},
        }

        # ---- EXTRACT ----
        logger.info("=" * 60)
        logger.info("STAGE 1: EXTRACT")
        logger.info("=" * 60)
        t0 = datetime.now()
        mock_api_data = self._prepare_synthetic_sources()
        raw_datasets = self.extractor.extract_all(mock_api_data)
        self.stage_times["extract"] = (datetime.now() - t0).total_seconds()
        report["stages"]["extract"] = {
            "duration_sec": self.stage_times["extract"],
            "stats": dict(self.extractor.stats),
        }

        # Map source names to logical names
        dataset_map = {
            "customers_csv": "customers",
            "orders_json": "orders",
            "products_api": "products",
        }
        datasets = {
            dataset_map.get(k, k): v for k, v in raw_datasets.items()
        }

        # Quality check on raw data
        logger.info("\n--- Raw Data Quality Checks ---")
        for name, data in datasets.items():
            required = self.config["transform"]["required_fields"].get(name, [])
            self.quality_checker.run_all_checks(name + "_raw", data, required)

        # ---- TRANSFORM ----
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: TRANSFORM")
        logger.info("=" * 60)
        t0 = datetime.now()
        transformed: dict[str, list[dict]] = {}
        # Transform products first (needed for order enrichment)
        products_clean, self.rejected["products"] = self.transformer.transform_dataset(
            "products", datasets.get("products", [])
        )
        transformed["products"] = products_clean

        # Transform customers
        cust_clean, self.rejected["customers"] = self.transformer.transform_dataset(
            "customers", datasets.get("customers", [])
        )
        transformed["customers"] = cust_clean

        # Transform orders (with product enrichment)
        orders_clean, self.rejected["orders"] = self.transformer.transform_dataset(
            "orders", datasets.get("orders", []), products=products_clean
        )
        transformed["orders"] = orders_clean

        self.stage_times["transform"] = (datetime.now() - t0).total_seconds()
        report["stages"]["transform"] = {
            "duration_sec": self.stage_times["transform"],
            "stats": dict(self.transformer.stats),
            "rejected_counts": {k: len(v) for k, v in self.rejected.items()},
        }

        # Quality check on transformed data
        logger.info("\n--- Transformed Data Quality Checks ---")
        for name, data in transformed.items():
            required = self.config["transform"]["required_fields"].get(name, [])
            self.quality_checker.run_all_checks(name + "_clean", data, required)

        # ---- LOAD ----
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: LOAD")
        logger.info("=" * 60)
        t0 = datetime.now()
        self.loader.load_all(transformed)

        # Also save rejected records
        for name, rej in self.rejected.items():
            if rej:
                self.loader.load_to_csv(rej, f"{name}_rejected.csv")

        self.stage_times["load"] = (datetime.now() - t0).total_seconds()
        report["stages"]["load"] = {
            "duration_sec": self.stage_times["load"],
            "stats": dict(self.loader.stats),
        }

        # ---- FINALIZE ----
        self.end_time = datetime.now()
        total_time = (self.end_time - self.start_time).total_seconds()
        report["end_time"] = self.end_time.isoformat()
        report["total_duration_sec"] = total_time
        report["quality_report"] = self.quality_checker.summary()

        return report

    def print_report(self, report: dict) -> None:
        """Print a formatted pipeline execution report."""
        print("\n" + "=" * 70)
        print("ETL PIPELINE EXECUTION REPORT")
        print("=" * 70)
        print(f"Pipeline     : {report['pipeline']} v{report['version']}")
        print(f"Start Time   : {report['start_time']}")
        print(f"End Time     : {report['end_time']}")
        print(f"Duration     : {report['total_duration_sec']:.3f} seconds")
        print(f"Working Dir  : {report['working_dir']}")

        print("\n--- Stage Durations ---")
        for stage, info in report["stages"].items():
            print(f"  {stage:12s}: {info['duration_sec']:.3f}s")

        print("\n--- Extract Stats ---")
        for source, stats in report["stages"]["extract"]["stats"].items():
            print(f"  {source}: {stats}")

        print("\n--- Transform Stats ---")
        for ds, stats in report["stages"]["transform"]["stats"].items():
            print(f"  {ds}: {stats}")
        print("  Rejected counts:", report["stages"]["transform"]["rejected_counts"])

        print("\n--- Load Stats ---")
        for target, stats in report["stages"]["load"]["stats"].items():
            print(f"  {target}: {stats}")

        print(report["quality_report"])

        # Verify SQLite tables
        db_path = os.path.join(
            self.working_dir,
            self.config["targets"]["sqlite"]["database"],
        )
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            print("\n--- SQLite Verification ---")
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            for (table_name,) in cursor:
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                print(f"  Table {table_name}: {count} rows")
            conn.close()

        print("\n" + "=" * 70)
        print("Pipeline completed successfully!")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("ETL Pipeline - Data Engineering Portfolio Project")
    print("-" * 50)
    pipeline = ETLPipeline(config=DEFAULT_CONFIG)
    report = pipeline.run()
    pipeline.print_report(report)


if __name__ == "__main__":
    main()
