"""
REST API Data Collection Framework
=====================================
Demonstrates API data engineering patterns with:
- Simulated REST API endpoints with mock responses
- Authentication handling (API key, OAuth mock)
- Pagination and rate limiting
- Data transformation and normalization
- Incremental data loading (track last fetched timestamp)
- Data storage with versioning
- Error handling and retry with exponential backoff
"""

import json
import csv
import os
import sys
import time
import random
import logging
import tempfile
import hashlib
import sqlite3
import copy
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Any, Optional
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("APICollector")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COLLECTOR_CONFIG = {
    "name": "multi_api_collector",
    "version": "1.0.0",
    "apis": {
        "users": {
            "base_url": "https://api.example.com/v1",
            "endpoint": "/users",
            "auth_type": "api_key",
            "api_key": "demo-key-abc123",
            "pagination": {"type": "offset", "page_size": 10},
            "rate_limit": {"requests_per_second": 5, "burst": 10},
        },
        "transactions": {
            "base_url": "https://api.example.com/v1",
            "endpoint": "/transactions",
            "auth_type": "oauth",
            "oauth_config": {
                "client_id": "demo-client",
                "client_secret": "demo-secret",
                "token_url": "https://auth.example.com/token",
            },
            "pagination": {"type": "cursor", "page_size": 20},
            "rate_limit": {"requests_per_second": 3, "burst": 5},
        },
        "products": {
            "base_url": "https://api.example.com/v1",
            "endpoint": "/products",
            "auth_type": "api_key",
            "api_key": "demo-key-abc123",
            "pagination": {"type": "page_number", "page_size": 15},
            "rate_limit": {"requests_per_second": 10, "burst": 20},
        },
    },
    "storage": {
        "type": "sqlite",
        "database": "api_data.db",
    },
    "incremental": {
        "state_file": "collector_state.json",
    },
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class APIResponse:
    """Represents an API response."""
    status_code: int
    body: dict
    headers: dict = field(default_factory=dict)
    request_url: str = ""
    request_time_ms: float = 0.0


@dataclass
class CollectionMetrics:
    """Tracks collection metrics."""
    api_name: str = ""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retries: int = 0
    records_fetched: int = 0
    records_stored: int = 0
    total_time_sec: float = 0.0
    rate_limit_waits: int = 0
    auth_refreshes: int = 0
    pages_fetched: int = 0

    def summary(self) -> str:
        lines = [
            f"\n  --- {self.api_name} Collection Metrics ---",
            f"    Total requests     : {self.total_requests}",
            f"    Successful         : {self.successful_requests}",
            f"    Failed             : {self.failed_requests}",
            f"    Retries            : {self.retries}",
            f"    Pages fetched      : {self.pages_fetched}",
            f"    Records fetched    : {self.records_fetched}",
            f"    Records stored     : {self.records_stored}",
            f"    Rate limit waits   : {self.rate_limit_waits}",
            f"    Auth refreshes     : {self.auth_refreshes}",
            f"    Total time         : {self.total_time_sec:.2f}s",
        ]
        if self.total_time_sec > 0:
            lines.append(
                f"    Throughput         : {self.records_fetched / self.total_time_sec:.1f} rec/sec"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mock API Server
# ---------------------------------------------------------------------------
class MockAPIServer:
    """Simulates REST API endpoints with realistic behavior."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.data = self._generate_data()
        self.tokens: dict[str, dict] = {}
        self.request_count = 0
        self.failure_rate = 0.05

    def _generate_data(self) -> dict:
        """Generate synthetic API data."""
        rng = self.rng

        # Users
        first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
                       "Grace", "Hank", "Iris", "Jack", "Karen", "Leo",
                       "Mona", "Nate", "Olivia", "Paul", "Quinn", "Rose",
                       "Steve", "Tina", "Uma", "Victor", "Wendy", "Xander",
                       "Yara", "Zane"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones",
                      "Garcia", "Miller", "Davis", "Wilson", "Anderson"]
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                  "Dallas", "San Jose", "Austin", "Seattle", "Denver"]

        users = []
        for i in range(50):
            first = rng.choice(first_names)
            last = rng.choice(last_names)
            created = datetime(2023, 1, 1) + timedelta(days=rng.randint(0, 730))
            updated = created + timedelta(days=rng.randint(0, 60))
            users.append({
                "id": i + 1,
                "username": f"{first.lower()}{last.lower()}{i}",
                "email": f"{first.lower()}.{last.lower()}{i}@example.com",
                "first_name": first,
                "last_name": last,
                "city": rng.choice(cities),
                "is_active": rng.random() > 0.1,
                "plan": rng.choice(["free", "basic", "premium", "enterprise"]),
                "created_at": created.isoformat() + "Z",
                "updated_at": updated.isoformat() + "Z",
            })

        # Transactions
        categories = ["Electronics", "Clothing", "Food", "Transport",
                      "Entertainment", "Health", "Education"]
        statuses = ["completed", "pending", "failed", "refunded"]
        transactions = []
        for i in range(200):
            ts = datetime(2024, 1, 1) + timedelta(
                hours=rng.randint(0, 8760)
            )
            transactions.append({
                "id": f"TXN-{i+1:06d}",
                "user_id": rng.randint(1, 50),
                "amount": round(rng.uniform(1.0, 500.0), 2),
                "currency": rng.choice(["USD", "EUR", "GBP"]),
                "category": rng.choice(categories),
                "status": rng.choices(statuses, weights=[70, 15, 5, 10])[0],
                "description": f"Transaction {i+1}",
                "timestamp": ts.isoformat() + "Z",
            })
        # Sort by timestamp for incremental loading
        transactions.sort(key=lambda t: t["timestamp"])

        # Products
        product_names = [
            "Widget A", "Widget B", "Gadget Pro", "Gadget Lite",
            "Module X1", "Module X2", "Sensor Array", "Controller Board",
            "Power Supply", "Display Unit", "Cable Kit", "Adapter Pack",
            "Battery Module", "Cooling Fan", "LED Panel", "Relay Switch",
            "Motor Drive", "Signal Converter", "Interface Card", "Connector Set",
            "Fuse Box", "Transformer", "Oscillator", "Amplifier", "Filter Unit",
            "Actuator", "Encoder", "Decoder", "Capacitor Set", "Resistor Pack",
        ]
        products = []
        for i, name in enumerate(product_names):
            products.append({
                "id": f"PROD-{i+1:04d}",
                "name": name,
                "category": rng.choice(["Components", "Modules", "Accessories", "Kits"]),
                "price": round(rng.uniform(5.0, 200.0), 2),
                "stock_quantity": rng.randint(0, 1000),
                "is_available": rng.random() > 0.15,
                "sku": f"SKU-{rng.randint(10000, 99999)}",
                "created_at": (datetime(2023, 6, 1) + timedelta(days=rng.randint(0, 365))).isoformat() + "Z",
            })

        return {
            "users": users,
            "transactions": transactions,
            "products": products,
        }

    def _maybe_fail(self) -> Optional[APIResponse]:
        """Randomly simulate API failures."""
        self.request_count += 1
        if self.rng.random() < self.failure_rate:
            code = self.rng.choice([429, 500, 502, 503])
            return APIResponse(
                status_code=code,
                body={"error": f"Server error {code}"},
                headers={"Retry-After": "1"} if code == 429 else {},
            )
        return None

    def authenticate_oauth(self, client_id: str, client_secret: str) -> APIResponse:
        """Mock OAuth token endpoint."""
        if client_id == "demo-client" and client_secret == "demo-secret":
            token = hashlib.md5(
                f"{client_id}:{time.time()}".encode()
            ).hexdigest()
            self.tokens[token] = {
                "client_id": client_id,
                "expires_at": time.time() + 3600,
            }
            return APIResponse(
                status_code=200,
                body={
                    "access_token": token,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        return APIResponse(
            status_code=401,
            body={"error": "invalid_client"},
        )

    def validate_auth(self, auth_type: str, credentials: dict) -> bool:
        """Validate request authentication."""
        if auth_type == "api_key":
            return credentials.get("api_key") == "demo-key-abc123"
        elif auth_type == "oauth":
            token = credentials.get("access_token", "")
            token_info = self.tokens.get(token)
            if token_info and token_info["expires_at"] > time.time():
                return True
        return False

    def handle_request(self, endpoint: str, params: dict,
                       auth_type: str, credentials: dict) -> APIResponse:
        """Handle a mock API request."""
        # Simulate failures
        failure = self._maybe_fail()
        if failure:
            return failure

        # Auth check
        if not self.validate_auth(auth_type, credentials):
            return APIResponse(status_code=401, body={"error": "Unauthorized"})

        # Route to handler
        t0 = time.time()
        if endpoint == "/users":
            response = self._handle_users(params)
        elif endpoint == "/transactions":
            response = self._handle_transactions(params)
        elif endpoint == "/products":
            response = self._handle_products(params)
        else:
            response = APIResponse(status_code=404, body={"error": "Not found"})

        response.request_time_ms = (time.time() - t0) * 1000
        return response

    def _handle_users(self, params: dict) -> APIResponse:
        """Handle /users endpoint with offset pagination."""
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", 10))
        since = params.get("updated_since")

        data = self.data["users"]
        if since:
            data = [u for u in data if u["updated_at"] > since]

        page_data = data[offset:offset + limit]
        return APIResponse(
            status_code=200,
            body={
                "data": page_data,
                "pagination": {
                    "offset": offset,
                    "limit": limit,
                    "total": len(data),
                    "has_more": offset + limit < len(data),
                },
            },
            headers={"X-Total-Count": str(len(data))},
        )

    def _handle_transactions(self, params: dict) -> APIResponse:
        """Handle /transactions endpoint with cursor pagination."""
        cursor = params.get("cursor")
        limit = int(params.get("limit", 20))
        since = params.get("since")

        data = self.data["transactions"]
        if since:
            data = [t for t in data if t["timestamp"] > since]

        # Cursor-based: cursor is the index
        start_idx = 0
        if cursor:
            try:
                start_idx = int(cursor)
            except ValueError:
                start_idx = 0

        page_data = data[start_idx:start_idx + limit]
        next_cursor = None
        if start_idx + limit < len(data):
            next_cursor = str(start_idx + limit)

        return APIResponse(
            status_code=200,
            body={
                "data": page_data,
                "pagination": {
                    "cursor": cursor,
                    "next_cursor": next_cursor,
                    "limit": limit,
                    "has_more": next_cursor is not None,
                    "total_available": len(data),
                },
            },
        )

    def _handle_products(self, params: dict) -> APIResponse:
        """Handle /products endpoint with page number pagination."""
        page = int(params.get("page", 1))
        per_page = int(params.get("per_page", 15))

        data = self.data["products"]
        total_pages = (len(data) + per_page - 1) // per_page
        start = (page - 1) * per_page
        page_data = data[start:start + per_page]

        return APIResponse(
            status_code=200,
            body={
                "data": page_data,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_pages": total_pages,
                    "total_items": len(data),
                    "has_more": page < total_pages,
                },
            },
        )


# ---------------------------------------------------------------------------
# Authentication Manager
# ---------------------------------------------------------------------------
class AuthManager:
    """Manages API authentication credentials."""

    def __init__(self, api_server: MockAPIServer):
        self.server = api_server
        self.tokens: dict[str, dict] = {}

    def get_credentials(self, api_name: str, api_config: dict) -> dict:
        """Get valid credentials for an API."""
        auth_type = api_config["auth_type"]
        if auth_type == "api_key":
            return {"api_key": api_config["api_key"]}
        elif auth_type == "oauth":
            return self._get_oauth_token(api_name, api_config["oauth_config"])
        return {}

    def _get_oauth_token(self, api_name: str, oauth_config: dict) -> dict:
        """Get or refresh an OAuth token."""
        existing = self.tokens.get(api_name)
        if existing and existing.get("expires_at", 0) > time.time() + 60:
            return {"access_token": existing["access_token"]}

        logger.info(f"  Requesting OAuth token for {api_name}...")
        response = self.server.authenticate_oauth(
            oauth_config["client_id"],
            oauth_config["client_secret"],
        )
        if response.status_code == 200:
            token_data = response.body
            self.tokens[api_name] = {
                "access_token": token_data["access_token"],
                "expires_at": time.time() + token_data["expires_in"],
            }
            logger.info(f"  OAuth token obtained (expires in {token_data['expires_in']}s)")
            return {"access_token": token_data["access_token"]}
        else:
            raise RuntimeError(f"OAuth authentication failed: {response.body}")

    def refresh_if_needed(self, api_name: str, api_config: dict) -> dict:
        """Refresh credentials if they are about to expire."""
        return self.get_credentials(api_name, api_config)


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_second: float, burst: int):
        self.rate = requests_per_second
        self.burst = burst
        self.tokens = float(burst)
        self.last_time = time.time()

    def acquire(self) -> float:
        now = time.time()
        elapsed = now - self.last_time
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_time = now

        if self.tokens >= 1:
            self.tokens -= 1
            return 0.0

        wait = (1 - self.tokens) / self.rate
        time.sleep(wait)
        self.tokens = 0
        self.last_time = time.time()
        return wait


# ---------------------------------------------------------------------------
# Retry Handler
# ---------------------------------------------------------------------------
class RetryHandler:
    """Retry with exponential backoff."""

    def __init__(self, max_retries: int = 3, base_delay: float = 0.1,
                 max_delay: float = 5.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.rng = random.Random(42)

    def should_retry(self, response: APIResponse, attempt: int) -> tuple[bool, float]:
        """Determine if we should retry and how long to wait."""
        if attempt >= self.max_retries:
            return False, 0.0

        retryable_codes = {429, 500, 502, 503, 504}
        if response.status_code not in retryable_codes:
            return False, 0.0

        # Exponential backoff with jitter
        delay = min(
            self.base_delay * (2 ** attempt),
            self.max_delay,
        )
        delay *= self.rng.uniform(0.5, 1.5)

        # Respect Retry-After header
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass

        return True, delay


# ---------------------------------------------------------------------------
# Data Transformer
# ---------------------------------------------------------------------------
class DataTransformer:
    """Transforms and normalizes API data."""

    def __init__(self):
        self.stats: dict[str, int] = defaultdict(int)

    def transform_users(self, raw_users: list[dict]) -> list[dict]:
        """Normalize user data."""
        transformed = []
        for user in raw_users:
            transformed.append({
                "user_id": user["id"],
                "username": user["username"],
                "email": user.get("email", "").lower().strip(),
                "full_name": f"{user.get('first_name', '')} {user.get('last_name', '')}".strip(),
                "city": user.get("city", ""),
                "is_active": bool(user.get("is_active", False)),
                "plan": user.get("plan", "free"),
                "created_at": user.get("created_at", ""),
                "updated_at": user.get("updated_at", ""),
                "collected_at": datetime.now().isoformat(),
            })
            self.stats["users_transformed"] += 1
        return transformed

    def transform_transactions(self, raw_transactions: list[dict]) -> list[dict]:
        """Normalize transaction data."""
        transformed = []
        for txn in raw_transactions:
            amount = txn.get("amount", 0)
            currency = txn.get("currency", "USD")
            # Normalize to USD
            fx_rates = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27}
            usd_amount = round(amount * fx_rates.get(currency, 1.0), 2)

            transformed.append({
                "transaction_id": txn["id"],
                "user_id": txn["user_id"],
                "amount": amount,
                "currency": currency,
                "amount_usd": usd_amount,
                "category": txn.get("category", "Other"),
                "status": txn.get("status", "unknown"),
                "description": txn.get("description", ""),
                "timestamp": txn.get("timestamp", ""),
                "collected_at": datetime.now().isoformat(),
            })
            self.stats["transactions_transformed"] += 1
        return transformed

    def transform_products(self, raw_products: list[dict]) -> list[dict]:
        """Normalize product data."""
        transformed = []
        for prod in raw_products:
            transformed.append({
                "product_id": prod["id"],
                "name": prod["name"],
                "category": prod.get("category", ""),
                "price": prod.get("price", 0.0),
                "stock_quantity": prod.get("stock_quantity", 0),
                "is_available": bool(prod.get("is_available", False)),
                "sku": prod.get("sku", ""),
                "created_at": prod.get("created_at", ""),
                "collected_at": datetime.now().isoformat(),
            })
            self.stats["products_transformed"] += 1
        return transformed

    def transform(self, api_name: str, data: list[dict]) -> list[dict]:
        """Route to the appropriate transformer."""
        if api_name == "users":
            return self.transform_users(data)
        elif api_name == "transactions":
            return self.transform_transactions(data)
        elif api_name == "products":
            return self.transform_products(data)
        return data


# ---------------------------------------------------------------------------
# Incremental State Manager
# ---------------------------------------------------------------------------
class IncrementalStateManager:
    """Tracks collection state for incremental loading."""

    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state: dict[str, dict] = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {}

    def save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def get_last_timestamp(self, api_name: str) -> Optional[str]:
        return self.state.get(api_name, {}).get("last_timestamp")

    def update_timestamp(self, api_name: str, timestamp: str):
        if api_name not in self.state:
            self.state[api_name] = {}
        self.state[api_name]["last_timestamp"] = timestamp
        self.state[api_name]["last_run"] = datetime.now().isoformat()
        self.save_state()

    def get_run_history(self, api_name: str) -> list[dict]:
        return self.state.get(api_name, {}).get("history", [])

    def record_run(self, api_name: str, metrics: CollectionMetrics):
        if api_name not in self.state:
            self.state[api_name] = {}
        if "history" not in self.state[api_name]:
            self.state[api_name]["history"] = []
        self.state[api_name]["history"].append({
            "timestamp": datetime.now().isoformat(),
            "records_fetched": metrics.records_fetched,
            "records_stored": metrics.records_stored,
            "duration_sec": round(metrics.total_time_sec, 2),
        })
        self.save_state()


# ---------------------------------------------------------------------------
# Versioned Data Storage
# ---------------------------------------------------------------------------
class VersionedDataStore:
    """SQLite-based storage with versioning and snapshots."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create storage tables."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS data_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_name TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                record_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                checksum TEXT,
                UNIQUE(api_name, version_number)
            );

            CREATE TABLE IF NOT EXISTS collected_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_name TEXT NOT NULL,
                version_id INTEGER NOT NULL,
                record_key TEXT NOT NULL,
                record_data TEXT NOT NULL,
                collected_at TEXT NOT NULL,
                FOREIGN KEY (version_id) REFERENCES data_versions(version_id)
            );

            CREATE INDEX IF NOT EXISTS idx_data_api
                ON collected_data(api_name, version_id);
            CREATE INDEX IF NOT EXISTS idx_data_key
                ON collected_data(record_key);
        """)
        self.conn.commit()

    def get_next_version(self, api_name: str) -> int:
        cursor = self.conn.execute(
            "SELECT MAX(version_number) FROM data_versions WHERE api_name=?",
            (api_name,),
        )
        row = cursor.fetchone()
        current = row[0] if row[0] is not None else 0
        return current + 1

    def store_batch(self, api_name: str, records: list[dict],
                    key_field: str) -> int:
        """Store a batch of records with versioning."""
        if not records:
            return 0

        version = self.get_next_version(api_name)

        # Compute checksum
        content = json.dumps(records, sort_keys=True, default=str)
        checksum = hashlib.md5(content.encode()).hexdigest()

        # Create version record
        self.conn.execute(
            "INSERT INTO data_versions (api_name, version_number, record_count, "
            "created_at, checksum) VALUES (?,?,?,?,?)",
            (api_name, version, len(records), datetime.now().isoformat(), checksum),
        )

        # Get version_id
        cursor = self.conn.execute(
            "SELECT version_id FROM data_versions WHERE api_name=? AND version_number=?",
            (api_name, version),
        )
        version_id = cursor.fetchone()["version_id"]

        # Insert records
        for record in records:
            record_key = str(record.get(key_field, ""))
            self.conn.execute(
                "INSERT INTO collected_data (api_name, version_id, record_key, "
                "record_data, collected_at) VALUES (?,?,?,?,?)",
                (api_name, version_id, record_key,
                 json.dumps(record, default=str), datetime.now().isoformat()),
            )

        self.conn.commit()
        logger.info(
            f"  Stored {len(records)} records for {api_name} (v{version}, "
            f"checksum: {checksum[:8]})"
        )
        return len(records)

    def get_version_history(self, api_name: str) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT * FROM data_versions WHERE api_name=? ORDER BY version_number",
            (api_name,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_latest_data(self, api_name: str) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT record_data FROM collected_data cd "
            "JOIN data_versions dv ON cd.version_id = dv.version_id "
            "WHERE cd.api_name=? AND dv.version_number = "
            "(SELECT MAX(version_number) FROM data_versions WHERE api_name=?)",
            (api_name, api_name),
        )
        return [json.loads(row["record_data"]) for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        stats = {}
        cursor = self.conn.execute(
            "SELECT api_name, COUNT(*) as versions, SUM(record_count) as total_records "
            "FROM data_versions GROUP BY api_name"
        )
        for row in cursor.fetchall():
            stats[row["api_name"]] = {
                "versions": row["versions"],
                "total_records": row["total_records"],
            }
        return stats

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# API Collector Engine
# ---------------------------------------------------------------------------
class APICollector:
    """Core engine for collecting data from REST APIs."""

    def __init__(self, config: dict, output_dir: Optional[str] = None):
        self.config = config
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="api_collect_")
        os.makedirs(self.output_dir, exist_ok=True)

        self.server = MockAPIServer(seed=42)
        self.auth_manager = AuthManager(self.server)
        self.transformer = DataTransformer()
        self.state_manager = IncrementalStateManager(
            os.path.join(self.output_dir, config["incremental"]["state_file"])
        )
        self.data_store = VersionedDataStore(
            os.path.join(self.output_dir, config["storage"]["database"])
        )
        self.rate_limiters: dict[str, RateLimiter] = {}
        self.retry_handler = RetryHandler(max_retries=3, base_delay=0.1, max_delay=2.0)
        self.all_metrics: dict[str, CollectionMetrics] = {}

    def _get_rate_limiter(self, api_name: str) -> RateLimiter:
        if api_name not in self.rate_limiters:
            rl_config = self.config["apis"][api_name]["rate_limit"]
            self.rate_limiters[api_name] = RateLimiter(
                rl_config["requests_per_second"],
                rl_config["burst"],
            )
        return self.rate_limiters[api_name]

    def _make_request(self, api_name: str, params: dict,
                      metrics: CollectionMetrics) -> Optional[APIResponse]:
        """Make a single API request with rate limiting, auth, and retries."""
        api_config = self.config["apis"][api_name]
        rate_limiter = self._get_rate_limiter(api_name)

        # Rate limiting
        wait = rate_limiter.acquire()
        if wait > 0:
            metrics.rate_limit_waits += 1

        # Auth
        credentials = self.auth_manager.refresh_if_needed(api_name, api_config)
        if api_config["auth_type"] == "oauth":
            metrics.auth_refreshes += 1

        # Request with retries
        for attempt in range(self.retry_handler.max_retries + 1):
            metrics.total_requests += 1
            response = self.server.handle_request(
                api_config["endpoint"], params,
                api_config["auth_type"], credentials,
            )

            if response.status_code == 200:
                metrics.successful_requests += 1
                return response

            should_retry, delay = self.retry_handler.should_retry(response, attempt)
            if should_retry:
                metrics.retries += 1
                logger.warning(
                    f"  Request failed ({response.status_code}), "
                    f"retry {attempt+1} in {delay:.2f}s"
                )
                time.sleep(delay)

                # Re-auth on 401
                if response.status_code == 401:
                    credentials = self.auth_manager.get_credentials(api_name, api_config)
                    metrics.auth_refreshes += 1
            else:
                metrics.failed_requests += 1
                logger.error(
                    f"  Request failed permanently ({response.status_code})"
                )
                return None

        metrics.failed_requests += 1
        return None

    def _collect_offset_paginated(self, api_name: str,
                                  metrics: CollectionMetrics,
                                  since: Optional[str] = None) -> list[dict]:
        """Collect data using offset-based pagination."""
        api_config = self.config["apis"][api_name]
        page_size = api_config["pagination"]["page_size"]
        all_records: list[dict] = []
        offset = 0

        while True:
            params = {"offset": str(offset), "limit": str(page_size)}
            if since:
                params["updated_since"] = since

            response = self._make_request(api_name, params, metrics)
            if not response:
                break

            records = response.body.get("data", [])
            all_records.extend(records)
            metrics.pages_fetched += 1
            metrics.records_fetched += len(records)

            pagination = response.body.get("pagination", {})
            if not pagination.get("has_more", False):
                break
            offset += page_size

        return all_records

    def _collect_cursor_paginated(self, api_name: str,
                                  metrics: CollectionMetrics,
                                  since: Optional[str] = None) -> list[dict]:
        """Collect data using cursor-based pagination."""
        api_config = self.config["apis"][api_name]
        page_size = api_config["pagination"]["page_size"]
        all_records: list[dict] = []
        cursor = None

        while True:
            params = {"limit": str(page_size)}
            if cursor:
                params["cursor"] = cursor
            if since:
                params["since"] = since

            response = self._make_request(api_name, params, metrics)
            if not response:
                break

            records = response.body.get("data", [])
            all_records.extend(records)
            metrics.pages_fetched += 1
            metrics.records_fetched += len(records)

            pagination = response.body.get("pagination", {})
            if not pagination.get("has_more", False):
                break
            cursor = pagination.get("next_cursor")
            if not cursor:
                break

        return all_records

    def _collect_page_number_paginated(self, api_name: str,
                                       metrics: CollectionMetrics) -> list[dict]:
        """Collect data using page number pagination."""
        api_config = self.config["apis"][api_name]
        page_size = api_config["pagination"]["page_size"]
        all_records: list[dict] = []
        page = 1

        while True:
            params = {"page": str(page), "per_page": str(page_size)}

            response = self._make_request(api_name, params, metrics)
            if not response:
                break

            records = response.body.get("data", [])
            all_records.extend(records)
            metrics.pages_fetched += 1
            metrics.records_fetched += len(records)

            pagination = response.body.get("pagination", {})
            if not pagination.get("has_more", False):
                break
            page += 1

        return all_records

    def collect_api(self, api_name: str) -> tuple[list[dict], CollectionMetrics]:
        """Collect all data from a single API."""
        logger.info(f"\nCollecting from API: {api_name}")
        api_config = self.config["apis"][api_name]
        metrics = CollectionMetrics(api_name=api_name)
        t0 = time.time()

        # Check for incremental loading
        since = self.state_manager.get_last_timestamp(api_name)
        if since:
            logger.info(f"  Incremental load since: {since}")

        # Route to correct pagination strategy
        pagination_type = api_config["pagination"]["type"]
        if pagination_type == "offset":
            raw_data = self._collect_offset_paginated(api_name, metrics, since)
        elif pagination_type == "cursor":
            raw_data = self._collect_cursor_paginated(api_name, metrics, since)
        elif pagination_type == "page_number":
            raw_data = self._collect_page_number_paginated(api_name, metrics)
        else:
            logger.error(f"  Unknown pagination type: {pagination_type}")
            raw_data = []

        # Transform
        transformed = self.transformer.transform(api_name, raw_data)

        # Store
        key_fields = {
            "users": "user_id",
            "transactions": "transaction_id",
            "products": "product_id",
        }
        stored = self.data_store.store_batch(
            api_name, transformed, key_fields.get(api_name, "id")
        )
        metrics.records_stored = stored

        # Update incremental state
        if transformed:
            timestamp_fields = {
                "users": "updated_at",
                "transactions": "timestamp",
                "products": "created_at",
            }
            ts_field = timestamp_fields.get(api_name, "collected_at")
            timestamps = [r.get(ts_field, "") for r in transformed if r.get(ts_field)]
            if timestamps:
                latest = max(timestamps)
                self.state_manager.update_timestamp(api_name, latest)

        metrics.total_time_sec = time.time() - t0
        self.state_manager.record_run(api_name, metrics)
        self.all_metrics[api_name] = metrics
        return transformed, metrics

    def collect_all(self) -> dict[str, list[dict]]:
        """Collect data from all configured APIs."""
        results: dict[str, list[dict]] = {}
        for api_name in self.config["apis"]:
            data, metrics = self.collect_api(api_name)
            results[api_name] = data
            logger.info(metrics.summary())
        return results

    def export_csv(self, results: dict[str, list[dict]]) -> dict[str, str]:
        """Export collected data to CSV files."""
        paths: dict[str, str] = {}
        for api_name, data in results.items():
            if not data:
                continue
            filepath = os.path.join(self.output_dir, f"{api_name}.csv")
            fields = list(data[0].keys())
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
            paths[api_name] = filepath
            logger.info(f"  Exported {len(data)} records to {filepath}")
        return paths

    def print_report(self, results: dict[str, list[dict]]):
        """Print collection report."""
        print("\n" + "=" * 70)
        print("API DATA COLLECTION REPORT")
        print("=" * 70)
        print(f"Collector  : {self.config['name']} v{self.config['version']}")
        print(f"Output Dir : {self.output_dir}")

        for api_name, metrics in self.all_metrics.items():
            print(metrics.summary())

        print("\n--- Transformation Stats ---")
        for k, v in self.transformer.stats.items():
            print(f"  {k}: {v}")

        print("\n--- Storage Stats ---")
        store_stats = self.data_store.get_stats()
        for api_name, stats in store_stats.items():
            print(f"  {api_name}: {stats['versions']} versions, {stats['total_records']} total records")

        print("\n--- Version History ---")
        for api_name in self.config["apis"]:
            history = self.data_store.get_version_history(api_name)
            for v in history:
                print(
                    f"  {api_name} v{v['version_number']}: "
                    f"{v['record_count']} records "
                    f"(checksum: {v['checksum'][:8] if v['checksum'] else 'N/A'}) "
                    f"at {v['created_at']}"
                )

        print("\n--- Incremental State ---")
        for api_name in self.config["apis"]:
            ts = self.state_manager.get_last_timestamp(api_name)
            print(f"  {api_name}: last_timestamp = {ts}")

        print("\n--- Data Samples ---")
        for api_name, data in results.items():
            print(f"\n  {api_name} ({len(data)} records):")
            for record in data[:3]:
                # Show just a few key fields
                keys = list(record.keys())[:5]
                sample = {k: record[k] for k in keys}
                print(f"    {sample}")
            if len(data) > 3:
                print(f"    ... and {len(data) - 3} more")

        # Summary
        total_records = sum(len(d) for d in results.values())
        total_requests = sum(m.total_requests for m in self.all_metrics.values())
        total_time = sum(m.total_time_sec for m in self.all_metrics.values())
        print(f"\n--- Overall Summary ---")
        print(f"  APIs collected     : {len(results)}")
        print(f"  Total records      : {total_records}")
        print(f"  Total requests     : {total_requests}")
        print(f"  Total time         : {total_time:.2f}s")
        print(f"  Database           : {self.data_store.db_path}")

        print("\n" + "=" * 70)
        print("API data collection completed successfully!")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("REST API Data Collector - Data Engineering Portfolio Project")
    print("-" * 58)

    collector = APICollector(config=COLLECTOR_CONFIG)

    # First run - full collection
    print("\n>>> FIRST RUN: Full collection")
    results = collector.collect_all()
    csv_paths = collector.export_csv(results)

    # Second run - incremental (simulates running again later)
    print("\n>>> SECOND RUN: Incremental collection")
    results2 = collector.collect_all()

    # Report
    collector.print_report(results)


if __name__ == "__main__":
    main()
