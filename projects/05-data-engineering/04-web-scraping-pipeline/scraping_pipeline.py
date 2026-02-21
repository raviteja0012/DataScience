"""
Web Scraping Pipeline - Architecture Demonstration
=====================================================
Demonstrates web scraping engineering concepts with:
- Simulated scraping pipeline with mock HTML responses
- HTML parsing with BeautifulSoup
- Rate limiting and retry logic
- Data extraction, cleaning, and storage
- Pagination handling
- Export to CSV and JSON
- Scraping ethics and robots.txt concepts
"""

import time
import random
import csv
import json
import os
import sys
import re
import logging
import tempfile
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Any, Optional
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ScrapingPipeline")

# ---------------------------------------------------------------------------
# Check for BeautifulSoup availability
# ---------------------------------------------------------------------------
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logger.info("BeautifulSoup not installed; using built-in HTML parser fallback")


# ---------------------------------------------------------------------------
# Fallback HTML Parser (when bs4 is not available)
# ---------------------------------------------------------------------------
class SimpleHTMLExtractor(HTMLParser):
    """Minimal HTML parser fallback for when BeautifulSoup is unavailable."""

    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._current_tag = None
        self._current_attrs: dict = {}
        self._current_data = ""
        self._in_target = False
        self._depth = 0
        self._item: dict = {}
        self._capture_stack: list[str] = []

    def reset_results(self):
        self.results = []
        self._item = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self._current_tag = tag
        self._current_attrs = attrs_dict
        cls = attrs_dict.get("class", "")
        if tag == "div" and "product-card" in cls:
            self._in_target = True
            self._item = {}
            self._depth = 0
        if self._in_target:
            self._depth += 1
            if tag == "h3" and "product-title" in cls:
                self._capture_stack.append("title")
            elif tag == "span" and "price" in cls:
                self._capture_stack.append("price")
            elif tag == "span" and "rating" in cls:
                self._capture_stack.append("rating")
            elif tag == "p" and "description" in cls:
                self._capture_stack.append("description")
            elif tag == "span" and "stock" in cls:
                self._capture_stack.append("stock")
            elif tag == "a":
                href = attrs_dict.get("href", "")
                if href and "product-link" in cls:
                    self._item["url"] = href

    def handle_endtag(self, tag):
        if self._in_target:
            self._depth -= 1
            if self._capture_stack and tag in ("h3", "span", "p", "a"):
                self._capture_stack.pop()
            if tag == "div" and self._depth <= 0:
                self._in_target = False
                if self._item:
                    self.results.append(dict(self._item))
                    self._item = {}

    def handle_data(self, data):
        text = data.strip()
        if text and self._capture_stack:
            field = self._capture_stack[-1]
            self._item[field] = text


# ---------------------------------------------------------------------------
# Mock HTML Generator
# ---------------------------------------------------------------------------
class MockHTMLGenerator:
    """Generates realistic mock HTML pages for scraping demonstration."""

    PRODUCT_NAMES = [
        "Wireless Bluetooth Headphones", "Ergonomic Office Chair",
        "4K Ultra HD Monitor", "Mechanical Gaming Keyboard",
        "USB-C Docking Station", "Noise Cancelling Earbuds",
        "Standing Desk Converter", "Portable SSD 1TB",
        "Smart LED Desk Lamp", "Webcam HD 1080p",
        "Laptop Cooling Pad", "Wireless Charging Pad",
        "Cable Management Box", "Monitor Light Bar",
        "Adjustable Laptop Stand", "Bluetooth Mouse",
        "Desk Organizer Set", "Blue Light Glasses",
        "Acoustic Foam Panels", "Ring Light with Stand",
    ]

    BRANDS = ["TechPro", "ComfortMax", "PixelView", "KeyMaster",
              "HubLink", "SoundWave", "DeskFlex", "SpeedStore"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.all_products = self._generate_product_data()

    def _generate_product_data(self) -> list[dict]:
        products = []
        for i, name in enumerate(self.PRODUCT_NAMES):
            products.append({
                "id": f"prod-{i+1:04d}",
                "name": name,
                "brand": self.rng.choice(self.BRANDS),
                "price": round(self.rng.uniform(19.99, 499.99), 2),
                "rating": round(self.rng.uniform(3.0, 5.0), 1),
                "reviews": self.rng.randint(10, 2500),
                "stock": self.rng.choice(["In Stock", "Low Stock", "Out of Stock",
                                          "In Stock", "In Stock", "In Stock"]),
                "description": f"High quality {name.lower()} with premium features. "
                               f"Brand: {self.rng.choice(self.BRANDS)}. "
                               f"Perfect for home and office use.",
                "url": f"/products/{i+1}",
            })
        return products

    def generate_product_listing_page(self, page: int = 1,
                                      per_page: int = 5) -> str:
        """Generate a mock product listing page with HTML."""
        start = (page - 1) * per_page
        end = start + per_page
        products = self.all_products[start:end]
        total_pages = (len(self.all_products) + per_page - 1) // per_page

        product_cards = []
        for p in products:
            card = f'''
            <div class="product-card" data-id="{p['id']}">
                <h3 class="product-title">
                    <a href="{p['url']}" class="product-link">{p['name']}</a>
                </h3>
                <span class="brand">{p['brand']}</span>
                <span class="price">${p['price']:.2f}</span>
                <span class="rating">{p['rating']}</span>
                <span class="reviews">({p['reviews']} reviews)</span>
                <p class="description">{p['description']}</p>
                <span class="stock {p['stock'].lower().replace(' ', '-')}">{p['stock']}</span>
            </div>'''
            product_cards.append(card)

        pagination_links = []
        for pg in range(1, total_pages + 1):
            active = ' class="active"' if pg == page else ""
            pagination_links.append(
                f'<a href="/products?page={pg}"{active}>{pg}</a>'
            )

        html = f"""<!DOCTYPE html>
<html>
<head><title>Product Listing - Page {page}</title></head>
<body>
    <div class="container">
        <h1>Product Catalog</h1>
        <div class="product-count">Showing {start+1}-{min(end, len(self.all_products))} of {len(self.all_products)}</div>
        <div class="product-grid">
            {''.join(product_cards)}
        </div>
        <div class="pagination">
            {''.join(pagination_links)}
        </div>
        <meta name="total-pages" content="{total_pages}">
    </div>
</body>
</html>"""
        return html

    def generate_product_detail_page(self, product_id: str) -> Optional[str]:
        """Generate a mock product detail page."""
        product = None
        for p in self.all_products:
            if p["id"] == product_id:
                product = p
                break
        if not product:
            return None

        specs = {
            "Weight": f"{self.rng.uniform(0.2, 5.0):.1f} kg",
            "Dimensions": f"{self.rng.randint(10,50)}x{self.rng.randint(10,40)}x{self.rng.randint(5,20)} cm",
            "Warranty": f"{self.rng.choice([1, 2, 3])} years",
            "Color": self.rng.choice(["Black", "White", "Silver", "Gray"]),
        }

        spec_rows = "".join(
            f'<tr><td class="spec-name">{k}</td><td class="spec-value">{v}</td></tr>'
            for k, v in specs.items()
        )

        return f"""<!DOCTYPE html>
<html>
<head><title>{product['name']} - Product Details</title></head>
<body>
    <div class="product-detail">
        <h1 class="product-name">{product['name']}</h1>
        <span class="brand-name">{product['brand']}</span>
        <div class="price-section">
            <span class="current-price">${product['price']:.2f}</span>
        </div>
        <div class="rating-section">
            <span class="avg-rating">{product['rating']}</span>
            <span class="review-count">{product['reviews']} reviews</span>
        </div>
        <div class="description-full">
            <p>{product['description']}</p>
        </div>
        <table class="specifications">
            {spec_rows}
        </table>
        <div class="availability">
            <span class="stock-status">{product['stock']}</span>
        </div>
    </div>
</body>
</html>"""

    def generate_robots_txt(self) -> str:
        return """# robots.txt for example-store.com
User-agent: *
Allow: /products
Allow: /categories
Disallow: /admin
Disallow: /checkout
Disallow: /user/
Crawl-delay: 2

User-agent: BadBot
Disallow: /

Sitemap: https://example-store.com/sitemap.xml
"""


# ---------------------------------------------------------------------------
# Robots.txt Parser
# ---------------------------------------------------------------------------
class RobotsTxtParser:
    """Parses robots.txt to determine crawling permissions."""

    def __init__(self, content: str, user_agent: str = "*"):
        self.rules: dict[str, dict] = defaultdict(lambda: {"allow": [], "disallow": [], "crawl_delay": None})
        self.user_agent = user_agent
        self._parse(content)

    def _parse(self, content: str):
        current_agents: list[str] = []
        for line in content.splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "user-agent":
                current_agents = [value]
            elif key == "allow":
                for agent in current_agents:
                    self.rules[agent]["allow"].append(value)
            elif key == "disallow":
                for agent in current_agents:
                    self.rules[agent]["disallow"].append(value)
            elif key == "crawl-delay":
                try:
                    delay = float(value)
                    for agent in current_agents:
                        self.rules[agent]["crawl_delay"] = delay
                except ValueError:
                    pass

    def is_allowed(self, path: str) -> bool:
        agent_rules = self.rules.get(self.user_agent, self.rules.get("*", {}))
        # Check disallow first
        for pattern in agent_rules.get("disallow", []):
            if path.startswith(pattern):
                # Check if there's a more specific allow
                for allow_pattern in agent_rules.get("allow", []):
                    if path.startswith(allow_pattern) and len(allow_pattern) > len(pattern):
                        return True
                return False
        return True

    def get_crawl_delay(self) -> Optional[float]:
        agent_rules = self.rules.get(self.user_agent, self.rules.get("*", {}))
        return agent_rules.get("crawl_delay")

    def summary(self) -> str:
        lines = ["Robots.txt Rules:"]
        for agent, rules in self.rules.items():
            lines.append(f"  User-agent: {agent}")
            for a in rules["allow"]:
                lines.append(f"    Allow: {a}")
            for d in rules["disallow"]:
                lines.append(f"    Disallow: {d}")
            if rules["crawl_delay"]:
                lines.append(f"    Crawl-delay: {rules['crawl_delay']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """Token bucket rate limiter for controlling request frequency."""

    def __init__(self, requests_per_second: float = 1.0, burst: int = 3):
        self.rate = requests_per_second
        self.burst = burst
        self.tokens = float(burst)
        self.last_time = time.time()
        self.request_times: list[float] = []

    def acquire(self) -> float:
        """Wait until a token is available. Returns wait time in seconds."""
        now = time.time()
        elapsed = now - self.last_time
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_time = now

        if self.tokens >= 1:
            self.tokens -= 1
            self.request_times.append(now)
            return 0.0

        wait_time = (1 - self.tokens) / self.rate
        time.sleep(wait_time)
        self.tokens = 0
        self.last_time = time.time()
        self.request_times.append(self.last_time)
        return wait_time

    @property
    def avg_rate(self) -> float:
        if len(self.request_times) < 2:
            return 0.0
        duration = self.request_times[-1] - self.request_times[0]
        if duration == 0:
            return 0.0
        return (len(self.request_times) - 1) / duration


# ---------------------------------------------------------------------------
# Retry Logic
# ---------------------------------------------------------------------------
class RetryHandler:
    """Implements retry logic with exponential backoff."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 30.0, jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.rng = random.Random(42)
        self.retry_stats: dict[str, int] = defaultdict(int)

    def execute_with_retry(self, func, *args, **kwargs) -> tuple[Any, int]:
        """Execute function with retries. Returns (result, attempts)."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return result, attempt + 1
            except Exception as e:
                last_exception = e
                self.retry_stats[type(e).__name__] += 1
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (2 ** attempt),
                        self.max_delay,
                    )
                    if self.jitter:
                        delay *= self.rng.uniform(0.5, 1.5)
                    logger.warning(
                        f"Attempt {attempt+1} failed ({e}), "
                        f"retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
        raise last_exception


# ---------------------------------------------------------------------------
# Mock HTTP Client
# ---------------------------------------------------------------------------
class MockHTTPClient:
    """Simulates HTTP requests with configurable failure rates."""

    def __init__(self, html_generator: MockHTMLGenerator,
                 failure_rate: float = 0.1, seed: int = 42):
        self.generator = html_generator
        self.failure_rate = failure_rate
        self.rng = random.Random(seed)
        self.request_count = 0
        self.request_log: list[dict] = []

    def get(self, url: str) -> dict:
        """Simulate an HTTP GET request."""
        self.request_count += 1
        entry = {
            "request_num": self.request_count,
            "url": url,
            "timestamp": datetime.now().isoformat(),
        }

        # Simulate random failures
        if self.rng.random() < self.failure_rate:
            status = self.rng.choice([429, 500, 503, 408])
            entry["status"] = status
            self.request_log.append(entry)
            raise ConnectionError(f"HTTP {status} for {url}")

        # Route to appropriate mock response
        parsed = urlparse(url)
        path = parsed.path
        query = parsed.query

        if path == "/robots.txt":
            body = self.generator.generate_robots_txt()
        elif path == "/products" or path == "/products/":
            page = 1
            if "page=" in query:
                page = int(query.split("page=")[1].split("&")[0])
            body = self.generator.generate_product_listing_page(page)
        elif path.startswith("/products/"):
            prod_num = path.split("/")[-1]
            prod_id = f"prod-{int(prod_num):04d}"
            body = self.generator.generate_product_detail_page(prod_id)
            if body is None:
                entry["status"] = 404
                self.request_log.append(entry)
                raise ConnectionError(f"HTTP 404 for {url}")
        else:
            entry["status"] = 404
            self.request_log.append(entry)
            raise ConnectionError(f"HTTP 404 for {url}")

        entry["status"] = 200
        entry["content_length"] = len(body)
        self.request_log.append(entry)
        return {"status": 200, "body": body, "url": url}


# ---------------------------------------------------------------------------
# HTML Data Extractor
# ---------------------------------------------------------------------------
class HTMLDataExtractor:
    """Extracts structured data from HTML using BeautifulSoup or fallback."""

    def __init__(self):
        self.extraction_stats: dict[str, int] = defaultdict(int)

    def extract_products_from_listing(self, html: str) -> list[dict]:
        """Extract product data from a listing page."""
        products = []
        if HAS_BS4:
            products = self._extract_with_bs4(html)
        else:
            products = self._extract_with_fallback(html)

        self.extraction_stats["products_extracted"] += len(products)
        return products

    def _extract_with_bs4(self, html: str) -> list[dict]:
        """Extract using BeautifulSoup."""
        soup = BeautifulSoup(html, "html.parser")
        products = []
        for card in soup.find_all("div", class_="product-card"):
            product = {}
            product["id"] = card.get("data-id", "")

            title_tag = card.find("h3", class_="product-title")
            if title_tag:
                link = title_tag.find("a")
                product["name"] = link.get_text(strip=True) if link else title_tag.get_text(strip=True)
                product["url"] = link.get("href", "") if link else ""

            brand_tag = card.find("span", class_="brand")
            product["brand"] = brand_tag.get_text(strip=True) if brand_tag else ""

            price_tag = card.find("span", class_="price")
            if price_tag:
                price_text = price_tag.get_text(strip=True).replace("$", "").replace(",", "")
                try:
                    product["price"] = float(price_text)
                except ValueError:
                    product["price"] = None

            rating_tag = card.find("span", class_="rating")
            if rating_tag:
                try:
                    product["rating"] = float(rating_tag.get_text(strip=True))
                except ValueError:
                    product["rating"] = None

            reviews_tag = card.find("span", class_="reviews")
            if reviews_tag:
                text = reviews_tag.get_text(strip=True)
                match = re.search(r"(\d+)", text)
                product["reviews"] = int(match.group(1)) if match else 0

            desc_tag = card.find("p", class_="description")
            product["description"] = desc_tag.get_text(strip=True) if desc_tag else ""

            stock_tag = card.find("span", class_="stock")
            product["stock"] = stock_tag.get_text(strip=True) if stock_tag else ""

            products.append(product)
        return products

    def _extract_with_fallback(self, html: str) -> list[dict]:
        """Extract using built-in HTML parser fallback."""
        parser = SimpleHTMLExtractor()
        parser.reset_results()
        parser.feed(html)
        products = []
        for raw in parser.results:
            product = {}
            product["name"] = raw.get("title", "")
            product["url"] = raw.get("url", "")
            price_text = raw.get("price", "").replace("$", "").replace(",", "")
            try:
                product["price"] = float(price_text)
            except (ValueError, TypeError):
                product["price"] = None
            try:
                product["rating"] = float(raw.get("rating", ""))
            except (ValueError, TypeError):
                product["rating"] = None
            product["description"] = raw.get("description", "")
            product["stock"] = raw.get("stock", "")
            product["brand"] = raw.get("brand", "")
            products.append(product)
        return products

    def extract_pagination_info(self, html: str) -> dict:
        """Extract pagination info from HTML."""
        if HAS_BS4:
            soup = BeautifulSoup(html, "html.parser")
            meta = soup.find("meta", attrs={"name": "total-pages"})
            total_pages = int(meta["content"]) if meta else 1
            active = soup.find("a", class_="active")
            current_page = int(active.get_text(strip=True)) if active else 1
        else:
            # Fallback: regex extraction
            match = re.search(r'name="total-pages"\s+content="(\d+)"', html)
            total_pages = int(match.group(1)) if match else 1
            match = re.search(r'class="active">(\d+)<', html)
            current_page = int(match.group(1)) if match else 1

        return {"current_page": current_page, "total_pages": total_pages}


# ---------------------------------------------------------------------------
# Data Cleaner
# ---------------------------------------------------------------------------
class DataCleaner:
    """Cleans and normalizes scraped data."""

    def __init__(self):
        self.stats: dict[str, int] = defaultdict(int)

    def clean_product(self, product: dict) -> dict:
        """Clean and validate a single product record."""
        cleaned = {}

        # Name
        name = str(product.get("name", "")).strip()
        cleaned["name"] = name
        if not name:
            self.stats["missing_name"] += 1

        # Price
        price = product.get("price")
        if isinstance(price, str):
            price = price.replace("$", "").replace(",", "").strip()
            try:
                price = float(price)
            except ValueError:
                price = None
        if price is not None and price < 0:
            price = abs(price)
            self.stats["negative_price_fixed"] += 1
        cleaned["price"] = price

        # Rating
        rating = product.get("rating")
        if rating is not None:
            try:
                rating = float(rating)
                rating = max(0.0, min(5.0, rating))
            except (ValueError, TypeError):
                rating = None
        cleaned["rating"] = rating

        # Brand
        cleaned["brand"] = str(product.get("brand", "")).strip()

        # Stock
        stock = str(product.get("stock", "")).strip()
        cleaned["stock"] = stock
        cleaned["in_stock"] = stock.lower() not in ("out of stock", "")

        # Description: remove extra whitespace
        desc = str(product.get("description", ""))
        cleaned["description"] = " ".join(desc.split())

        # URL
        cleaned["url"] = str(product.get("url", "")).strip()

        # Reviews
        cleaned["reviews"] = product.get("reviews", 0)

        # ID
        cleaned["id"] = product.get("id", "")

        # Metadata
        cleaned["scraped_at"] = datetime.now().isoformat()
        content = f"{cleaned['name']}|{cleaned['price']}|{cleaned['url']}"
        cleaned["content_hash"] = hashlib.md5(content.encode()).hexdigest()[:12]

        self.stats["total_cleaned"] += 1
        return cleaned

    def deduplicate(self, products: list[dict]) -> list[dict]:
        """Remove duplicate products based on content hash."""
        seen: set[str] = set()
        deduped: list[dict] = []
        for p in products:
            h = p.get("content_hash", "")
            if h not in seen:
                seen.add(h)
                deduped.append(p)
            else:
                self.stats["duplicates_removed"] += 1
        return deduped


# ---------------------------------------------------------------------------
# Data Storage
# ---------------------------------------------------------------------------
class DataStorage:
    """Handles data export to CSV and JSON."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_to_csv(self, data: list[dict], filename: str) -> str:
        if not data:
            return ""
        filepath = os.path.join(self.output_dir, filename)
        fields = list(data[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Saved {len(data)} records to {filepath}")
        return filepath

    def save_to_json(self, data: list[dict], filename: str) -> str:
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved {len(data)} records to {filepath}")
        return filepath


# ---------------------------------------------------------------------------
# Scraping Pipeline Orchestrator
# ---------------------------------------------------------------------------
class ScrapingPipeline:
    """Orchestrates the full web scraping pipeline."""

    def __init__(self, base_url: str = "https://example-store.com",
                 output_dir: Optional[str] = None):
        self.base_url = base_url
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="scrape_")
        self.html_gen = MockHTMLGenerator(seed=42)
        self.http_client = MockHTTPClient(self.html_gen, failure_rate=0.08)
        self.rate_limiter = RateLimiter(requests_per_second=2.0, burst=3)
        self.retry_handler = RetryHandler(max_retries=3, base_delay=0.1, max_delay=2.0)
        self.extractor = HTMLDataExtractor()
        self.cleaner = DataCleaner()
        self.storage = DataStorage(self.output_dir)
        self.all_products: list[dict] = []
        self.metrics: dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "pages_scraped": 0,
            "products_found": 0,
            "products_after_cleaning": 0,
            "total_requests": 0,
            "failed_requests": 0,
            "total_wait_time": 0.0,
        }

    def _fetch_page(self, url: str) -> dict:
        """Fetch a page with rate limiting and retry."""
        wait = self.rate_limiter.acquire()
        self.metrics["total_wait_time"] += wait
        self.metrics["total_requests"] += 1

        try:
            response, attempts = self.retry_handler.execute_with_retry(
                self.http_client.get, url
            )
            if attempts > 1:
                logger.info(f"  Succeeded after {attempts} attempts: {url}")
            return response
        except Exception as e:
            self.metrics["failed_requests"] += 1
            logger.error(f"  Failed to fetch {url}: {e}")
            raise

    def _check_robots_txt(self) -> RobotsTxtParser:
        """Fetch and parse robots.txt."""
        logger.info("Checking robots.txt...")
        response = self._fetch_page(f"{self.base_url}/robots.txt")
        parser = RobotsTxtParser(response["body"])
        print(parser.summary())
        delay = parser.get_crawl_delay()
        if delay:
            logger.info(f"  Respecting crawl-delay: {delay}s")
            self.rate_limiter = RateLimiter(
                requests_per_second=1.0 / delay, burst=1
            )
        return parser

    def _scrape_listing_pages(self, robots: RobotsTxtParser) -> list[dict]:
        """Scrape all product listing pages with pagination."""
        all_raw_products: list[dict] = []
        page = 1
        total_pages = None

        while True:
            url = f"{self.base_url}/products?page={page}"
            path = f"/products?page={page}"

            if not robots.is_allowed("/products"):
                logger.warning(f"  Robots.txt disallows: {path}")
                break

            logger.info(f"Scraping listing page {page}...")
            try:
                response = self._fetch_page(url)
            except Exception:
                logger.error(f"  Skipping page {page}")
                page += 1
                if total_pages and page > total_pages:
                    break
                continue

            html = response["body"]

            # Extract pagination info
            if total_pages is None:
                page_info = self.extractor.extract_pagination_info(html)
                total_pages = page_info["total_pages"]
                logger.info(f"  Total pages: {total_pages}")

            # Extract products
            products = self.extractor.extract_products_from_listing(html)
            logger.info(f"  Found {len(products)} products on page {page}")
            all_raw_products.extend(products)
            self.metrics["pages_scraped"] += 1

            page += 1
            if page > total_pages:
                break

        self.metrics["products_found"] = len(all_raw_products)
        return all_raw_products

    def run(self) -> dict:
        """Execute the full scraping pipeline."""
        self.metrics["start_time"] = datetime.now().isoformat()
        logger.info("=" * 60)
        logger.info("WEB SCRAPING PIPELINE")
        logger.info(f"Target: {self.base_url}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 60)

        # Step 1: Check robots.txt
        print("\n--- Step 1: Robots.txt Check ---")
        robots = self._check_robots_txt()

        # Step 2: Ethics notice
        print("\n--- Scraping Ethics Notice ---")
        print("  - Respecting robots.txt directives")
        print("  - Rate limiting requests to be polite")
        print("  - Using retry with exponential backoff")
        print("  - Identifying as a responsible scraper")
        print("  - Only collecting publicly available data")
        print("  - This demo uses mock data, no real sites are scraped")

        # Step 3: Scrape listing pages
        print("\n--- Step 2: Scraping Product Listings ---")
        raw_products = self._scrape_listing_pages(robots)

        # Step 4: Clean data
        print("\n--- Step 3: Cleaning and Processing Data ---")
        cleaned_products = []
        for raw in raw_products:
            cleaned = self.cleaner.clean_product(raw)
            cleaned_products.append(cleaned)

        # Deduplicate
        cleaned_products = self.cleaner.deduplicate(cleaned_products)
        self.metrics["products_after_cleaning"] = len(cleaned_products)
        logger.info(f"  Cleaned: {len(cleaned_products)} unique products")

        # Step 5: Export
        print("\n--- Step 4: Exporting Data ---")
        csv_path = self.storage.save_to_csv(cleaned_products, "products.csv")
        json_path = self.storage.save_to_json(cleaned_products, "products.json")

        # Save scraping metadata
        self.metrics["end_time"] = datetime.now().isoformat()
        self.metrics["http_request_log"] = self.http_client.request_log
        self.metrics["retry_stats"] = dict(self.retry_handler.retry_stats)
        self.metrics["cleaning_stats"] = dict(self.cleaner.stats)
        self.metrics["extraction_stats"] = dict(self.extractor.extraction_stats)
        meta_path = self.storage.save_to_json(self.metrics, "scraping_metadata.json")

        self.all_products = cleaned_products

        return {
            "products": cleaned_products,
            "csv_path": csv_path,
            "json_path": json_path,
            "metadata_path": meta_path,
            "metrics": self.metrics,
        }

    def print_report(self, results: dict):
        """Print pipeline execution report."""
        m = results["metrics"]
        print("\n" + "=" * 70)
        print("SCRAPING PIPELINE REPORT")
        print("=" * 70)
        print(f"Target URL         : {self.base_url}")
        print(f"Output directory   : {self.output_dir}")
        print(f"Start time         : {m['start_time']}")
        print(f"End time           : {m['end_time']}")

        print(f"\n--- Scraping Stats ---")
        print(f"  Pages scraped          : {m['pages_scraped']}")
        print(f"  Products found (raw)   : {m['products_found']}")
        print(f"  Products (cleaned)     : {m['products_after_cleaning']}")
        print(f"  Total HTTP requests    : {m['total_requests']}")
        print(f"  Failed requests        : {m['failed_requests']}")
        print(f"  Rate limit wait time   : {m['total_wait_time']:.2f}s")
        print(f"  Avg request rate       : {self.rate_limiter.avg_rate:.1f} req/sec")

        print(f"\n--- Cleaning Stats ---")
        for k, v in m.get("cleaning_stats", {}).items():
            print(f"  {k}: {v}")

        print(f"\n--- Retry Stats ---")
        for k, v in m.get("retry_stats", {}).items():
            print(f"  {k}: {v}")

        print(f"\n--- Sample Products ---")
        for p in results["products"][:5]:
            print(
                f"  {p.get('name', 'N/A'):40s} | "
                f"${p.get('price', 0):>8.2f} | "
                f"Rating: {p.get('rating', 'N/A')} | "
                f"Stock: {p.get('stock', 'N/A')}"
            )
        if len(results["products"]) > 5:
            print(f"  ... and {len(results['products']) - 5} more")

        print(f"\n--- Output Files ---")
        print(f"  CSV     : {results['csv_path']}")
        print(f"  JSON    : {results['json_path']}")
        print(f"  Metadata: {results['metadata_path']}")

        print("\n" + "=" * 70)
        print("Scraping pipeline completed successfully!")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Web Scraping Pipeline - Data Engineering Portfolio Project")
    print("-" * 55)

    pipeline = ScrapingPipeline(
        base_url="https://example-store.com",
    )
    results = pipeline.run()
    pipeline.print_report(results)


if __name__ == "__main__":
    main()
