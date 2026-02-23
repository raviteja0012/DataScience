"""Synthetic data generators for demonstration and testing.

Produces realistic-looking customer, product, and transaction records that
mimic the schema shapes of the legacy Oracle and SQL Server source systems
described in the project configuration.  Records intentionally include the
kinds of quality issues called out in the source-system YAML files (phone
format inconsistencies, partial NULLs, duplicate near-matches, etc.) so
that the identity resolution and reconciliation modules have meaningful
work to do.
"""

from __future__ import annotations

import hashlib
import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Reference data pools
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen", "Charles",
    "Lisa", "Daniel", "Nancy", "Matthew", "Betty", "Anthony", "Margaret",
    "Mark", "Sandra", "Donald", "Ashley", "Steven", "Dorothy", "Paul", "Kimberly",
    "Andrew", "Emily", "Joshua", "Donna", "Kenneth", "Michelle", "Kevin", "Carol",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
]

_COMPANY_NAMES = [
    "Apex Industries", "Vertex Solutions", "Pinnacle Systems", "Summit Corp",
    "Crestline Holdings", "Meridian Technologies", "Cascade Enterprises",
    "Horizon Manufacturing", "Vanguard Services", "Titan Logistics",
    "Atlas Equipment", "Zenith Electronics", "Stratos Consulting",
    "Forge Industrial", "Nexus Trading", "Prism Analytics",
    "Quantum Dynamics", "Sterling Partners", "Cobalt Engineering",
    "Ember Creative", "Ironclad Security", "Lighthouse Capital",
]

_STREETS = [
    "Main St", "Oak Ave", "Park Blvd", "Elm St", "Cedar Ln",
    "Maple Dr", "Washington Ave", "Lake Rd", "Hill St", "River Rd",
    "Pine St", "Walnut Ave", "Broadway", "5th Ave", "Market St",
    "Industrial Pkwy", "Commerce Dr", "Technology Way", "Innovation Blvd",
]

_CITIES = [
    ("New York", "NY", "10001"), ("Los Angeles", "CA", "90001"),
    ("Chicago", "IL", "60601"), ("Houston", "TX", "77001"),
    ("Phoenix", "AZ", "85001"), ("Philadelphia", "PA", "19101"),
    ("San Antonio", "TX", "78201"), ("San Diego", "CA", "92101"),
    ("Dallas", "TX", "75201"), ("San Jose", "CA", "95101"),
    ("Austin", "TX", "73301"), ("Jacksonville", "FL", "32099"),
    ("Columbus", "OH", "43004"), ("Charlotte", "NC", "28201"),
    ("Indianapolis", "IN", "46201"), ("Denver", "CO", "80201"),
    ("Seattle", "WA", "98101"), ("Boston", "MA", "02101"),
    ("Portland", "OR", "97201"), ("Atlanta", "GA", "30301"),
]

_PRODUCT_ADJECTIVES = [
    "Premium", "Standard", "Professional", "Industrial", "Heavy-Duty",
    "Compact", "Ultra", "Economy", "Deluxe", "Advanced",
]

_PRODUCT_NOUNS = [
    "Widget", "Bracket", "Connector", "Sensor", "Module", "Assembly",
    "Controller", "Valve", "Filter", "Bearing", "Actuator", "Regulator",
    "Capacitor", "Transformer", "Converter", "Adapter", "Coupler", "Relay",
]

_CATEGORIES = ["ELEC", "MECH", "HYDR", "PNEU", "CTRL", "SNSR", "SAFE"]

_PHONE_FORMATS = [
    "({}){}-{}",
    "{}-{}-{}",
    "+1{}{}{}",
    "({}) {}-{}",
    "{}.{}.{}",
]

_SHIP_METHODS = ["Ground", "Express", "Overnight", "Freight", "Economy"]


# ---------------------------------------------------------------------------
# Data classes representing generated rows
# ---------------------------------------------------------------------------

@dataclass
class OracleCustomer:
    CUST_ID: int
    CUST_NM: str
    CUST_ADDR_1: str
    CUST_ADDR_2: Optional[str]
    CUST_CITY: str
    CUST_ST: str
    CUST_ZIP: str
    CUST_CNTRY: str
    CUST_EMAIL: Optional[str]
    CUST_PHONE: Optional[str]
    TAX_ID: Optional[str]
    CUST_TYPE: str
    CUST_STATUS: str
    CREATED_DT: str
    MODIFIED_DT: Optional[str]
    CREATED_BY: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SqlServerCustomer:
    CustomerID: int
    FirstName: Optional[str]
    LastName: Optional[str]
    CompanyName: Optional[str]
    EmailAddress: Optional[str]
    PhoneNumber: Optional[str]
    AddressLine1: Optional[str]
    AddressLine2: Optional[str]
    City: Optional[str]
    StateProvince: Optional[str]
    PostalCode: Optional[str]
    CountryCode: Optional[str]
    TaxIdentifier: Optional[str]
    CustomerType: str
    Status: str
    CreatedDate: str
    ModifiedDate: Optional[str]
    CreatedBy: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class OracleProduct:
    PROD_ID: int
    PROD_SKU: str
    PROD_DESC: str
    PROD_CAT_CD: str
    UNIT_PRICE: float
    UNIT_COST: Optional[float]
    WEIGHT_KG: Optional[float]
    IS_ACTIVE: str
    CREATED_DT: str
    MODIFIED_DT: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SqlServerProduct:
    ProductID: int
    SKU: str
    ProductName: str
    CategoryCode: str
    ListPrice: float
    StandardCost: Optional[float]
    WeightKg: Optional[float]
    IsActive: int
    CreatedDate: str
    ModifiedDate: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class OracleOrder:
    ORDER_ID: int
    CUST_ID: int
    ORDER_DT: str
    SHIP_DT: Optional[str]
    ORDER_STATUS: str
    ORDER_TOTAL: float
    TAX_AMT: Optional[float]
    DISCOUNT_AMT: Optional[float]
    SHIP_METHOD: Optional[str]
    CREATED_DT: str

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class OracleOrderDetail:
    ORDER_DTL_ID: int
    ORDER_ID: int
    PROD_ID: int
    QTY: int
    UNIT_PRICE: float
    LINE_TOTAL: float
    DISCOUNT_PCT: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SqlServerOrder:
    SalesOrderID: int
    CustomerID: int
    OrderDate: str
    ShipDate: Optional[str]
    OrderStatus: str
    TotalAmount: float
    TaxAmount: Optional[float]
    DiscountAmount: Optional[float]
    ShippingMethod: Optional[str]
    CreatedDate: str

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SqlServerOrderDetail:
    SalesOrderDetailID: int
    SalesOrderID: int
    ProductID: int
    Quantity: int
    UnitPrice: float
    LineTotal: float
    DiscountPercent: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _random_date(start_year: int = 2015, end_year: int = 2024) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    dt = start + timedelta(days=random.randint(0, delta))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _random_phone() -> str:
    area = str(random.randint(200, 999))
    prefix = str(random.randint(200, 999))
    line = str(random.randint(1000, 9999))
    fmt = random.choice(_PHONE_FORMATS)
    return fmt.format(area, prefix, line)


def _random_tax_id() -> Optional[str]:
    if random.random() < 0.12:  # ~12% NULL rate per known issues
        return None
    return f"{random.randint(10, 99)}-{random.randint(1000000, 9999999)}"


def _random_email(first: str, last: str) -> Optional[str]:
    if random.random() < 0.05:
        return None
    domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com",
               "acme.org", "betatech.io", "corp.net"]
    sep = random.choice([".", "_", ""])
    return f"{first.lower()}{sep}{last.lower()}@{random.choice(domains)}"


def _random_sku(prefix: str, idx: int) -> str:
    return f"{prefix}-{idx:05d}-{random.choice(string.ascii_uppercase)}{random.randint(0,9)}"


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """Generates correlated synthetic records for Oracle and SQL Server sources.

    A configurable percentage of records are intentional *cross-source
    duplicates*: the same logical customer appears in both systems with
    slight variations (different formatting, typos, etc.) to exercise the
    identity resolution engine.
    """

    def __init__(
        self,
        num_customers: int = 500,
        num_products: int = 100,
        num_orders: int = 1000,
        cross_source_overlap_pct: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.num_customers = num_customers
        self.num_products = num_products
        self.num_orders = num_orders
        self.overlap_pct = cross_source_overlap_pct
        self._rng = random.Random(seed)
        random.seed(seed)

        # Shared identity pool for cross-source overlaps
        self._shared_identities: List[Dict[str, Any]] = []

    # -- Customer generation --------------------------------------------------

    def _generate_base_identity(self) -> Dict[str, Any]:
        first = random.choice(_FIRST_NAMES)
        last = random.choice(_LAST_NAMES)
        city, state, zip_code = random.choice(_CITIES)
        is_business = random.random() < 0.35
        company = random.choice(_COMPANY_NAMES) if is_business else None
        return {
            "first": first,
            "last": last,
            "company": company,
            "email": _random_email(first, last),
            "phone": _random_phone(),
            "addr1": f"{random.randint(100, 9999)} {random.choice(_STREETS)}",
            "addr2": f"Suite {random.randint(1, 500)}" if random.random() < 0.3 else None,
            "city": city,
            "state": state,
            "zip": zip_code,
            "country": "USA",
            "tax_id": _random_tax_id(),
            "is_business": is_business,
            "created": _random_date(),
        }

    def _identity_to_oracle(self, idx: int, ident: Dict[str, Any],
                            introduce_noise: bool = False) -> OracleCustomer:
        name = ident["company"] if ident["is_business"] else f"{ident['first']} {ident['last']}"
        if introduce_noise and random.random() < 0.3:
            # Introduce slight variation: extra space, different case
            name = name.upper() if random.random() < 0.5 else f" {name} "

        email = ident["email"]
        if introduce_noise and email and random.random() < 0.2:
            email = email.upper()

        phone = ident["phone"]
        if introduce_noise and phone:
            phone = _random_phone()  # different format

        return OracleCustomer(
            CUST_ID=idx,
            CUST_NM=name,
            CUST_ADDR_1=ident["addr1"],
            CUST_ADDR_2=ident["addr2"],
            CUST_CITY=ident["city"],
            CUST_ST=ident["state"],
            CUST_ZIP=ident["zip"],
            CUST_CNTRY="USA",
            CUST_EMAIL=email,
            CUST_PHONE=phone,
            TAX_ID=ident["tax_id"],
            CUST_TYPE="B" if ident["is_business"] else "I",
            CUST_STATUS=random.choice(["A", "A", "A", "I"]),
            CREATED_DT=ident["created"],
            MODIFIED_DT=_random_date(2023, 2024) if random.random() < 0.6 else None,
            CREATED_BY=random.choice(["admin", "etl_user", "migration", None]),
        )

    def _identity_to_sqlserver(self, idx: int, ident: Dict[str, Any],
                               introduce_noise: bool = False) -> SqlServerCustomer:
        first = ident["first"]
        last = ident["last"]
        if introduce_noise:
            if random.random() < 0.15:
                first = first[0] + "." if random.random() < 0.5 else first + " "
            if random.random() < 0.1:
                # Swap a character (typo)
                last = last[:-1] + random.choice(string.ascii_lowercase)

        email = ident["email"]
        if introduce_noise and email and random.random() < 0.08:
            # Use a "fake" looking email
            email = f"test_{random.randint(1,999)}@example.com"

        country_2 = "US"  # SQL Server uses alpha-2

        return SqlServerCustomer(
            CustomerID=idx,
            FirstName=first if not ident["is_business"] else None,
            LastName=last if not ident["is_business"] else ident["company"],
            CompanyName=ident["company"],
            EmailAddress=email,
            PhoneNumber=ident["phone"] if random.random() > 0.05 else None,
            AddressLine1=ident["addr1"],
            AddressLine2=ident["addr2"],
            City=ident["city"],
            StateProvince=ident["state"],
            PostalCode=ident["zip"],
            CountryCode=country_2,
            TaxIdentifier=ident["tax_id"],
            CustomerType="Business" if ident["is_business"] else "Individual",
            Status=random.choice(["Active", "Active", "Active", "Inactive", "Suspended"]),
            CreatedDate=ident["created"],
            ModifiedDate=_random_date(2023, 2024) if random.random() < 0.7 else None,
            CreatedBy=random.choice(["admin", "crm_sync", "import", None]),
        )

    def generate_customers(
        self,
    ) -> Tuple[List[OracleCustomer], List[SqlServerCustomer]]:
        """Return (oracle_customers, sqlserver_customers) with controlled overlap."""
        oracle_only = int(self.num_customers * (1 - self.overlap_pct))
        sql_only = int(self.num_customers * (1 - self.overlap_pct))
        overlap = int(self.num_customers * self.overlap_pct)

        oracle_custs: List[OracleCustomer] = []
        sql_custs: List[SqlServerCustomer] = []

        ora_id = 1000
        sql_id = 5000

        # Oracle-only customers
        for _ in range(oracle_only):
            ident = self._generate_base_identity()
            oracle_custs.append(self._identity_to_oracle(ora_id, ident))
            ora_id += 1

        # SQL Server-only customers
        for _ in range(sql_only):
            ident = self._generate_base_identity()
            sql_custs.append(self._identity_to_sqlserver(sql_id, ident))
            sql_id += 1

        # Cross-source overlapping customers
        for _ in range(overlap):
            ident = self._generate_base_identity()
            self._shared_identities.append(ident)
            oracle_custs.append(
                self._identity_to_oracle(ora_id, ident, introduce_noise=True)
            )
            sql_custs.append(
                self._identity_to_sqlserver(sql_id, ident, introduce_noise=True)
            )
            ora_id += 1
            sql_id += 1

        random.shuffle(oracle_custs)
        random.shuffle(sql_custs)
        return oracle_custs, sql_custs

    # -- Product generation ---------------------------------------------------

    def generate_products(
        self,
    ) -> Tuple[List[OracleProduct], List[SqlServerProduct]]:
        oracle_prods: List[OracleProduct] = []
        sql_prods: List[SqlServerProduct] = []

        ora_id = 100
        sql_id = 3000

        for i in range(self.num_products):
            adj = random.choice(_PRODUCT_ADJECTIVES)
            noun = random.choice(_PRODUCT_NOUNS)
            cat = random.choice(_CATEGORIES)
            price = round(random.uniform(5.0, 2500.0), 2)
            cost = round(price * random.uniform(0.3, 0.8), 2)
            weight = round(random.uniform(0.1, 50.0), 3) if random.random() < 0.8 else None
            sku = _random_sku(cat, i)
            created = _random_date(2010, 2022)
            active = random.random() < 0.85

            # 70% of products exist only in one system; 30% shared
            in_oracle = random.random() < 0.65 or i < 20
            in_sql = random.random() < 0.65 or i < 20

            if in_oracle:
                oracle_prods.append(OracleProduct(
                    PROD_ID=ora_id,
                    PROD_SKU=sku,
                    PROD_DESC=f"{adj} {noun} ({cat})",
                    PROD_CAT_CD=cat,
                    UNIT_PRICE=price,
                    UNIT_COST=cost if random.random() < 0.9 else None,
                    WEIGHT_KG=weight,
                    IS_ACTIVE="Y" if active else "N",
                    CREATED_DT=created,
                    MODIFIED_DT=_random_date(2023, 2024) if random.random() < 0.5 else None,
                ))
                ora_id += 1

            if in_sql:
                sql_prods.append(SqlServerProduct(
                    ProductID=sql_id,
                    SKU=sku if in_oracle and random.random() < 0.7 else _random_sku(cat, i + 5000),
                    ProductName=f"{adj} {noun}",
                    CategoryCode=cat,
                    ListPrice=price + (round(random.uniform(-2, 2), 2) if in_oracle else 0),
                    StandardCost=cost,
                    WeightKg=weight,
                    IsActive=1 if active else 0,
                    CreatedDate=created,
                    ModifiedDate=_random_date(2023, 2024) if random.random() < 0.5 else None,
                ))
                sql_id += 1

        return oracle_prods, sql_prods

    # -- Order generation -----------------------------------------------------

    def generate_orders(
        self,
        oracle_customers: List[OracleCustomer],
        sql_customers: List[SqlServerCustomer],
        oracle_products: List[OracleProduct],
        sql_products: List[SqlServerProduct],
    ) -> Tuple[
        List[OracleOrder], List[OracleOrderDetail],
        List[SqlServerOrder], List[SqlServerOrderDetail],
    ]:
        ora_orders: List[OracleOrder] = []
        ora_details: List[OracleOrderDetail] = []
        sql_orders: List[SqlServerOrder] = []
        sql_details: List[SqlServerOrderDetail] = []

        ora_order_id = 200000
        ora_detail_id = 900000
        sql_order_id = 600000
        sql_detail_id = 1800000

        statuses_ora = ["OP", "CL", "CL", "CL", "CN", "RT", "HD"]
        statuses_sql = ["Open", "Closed", "Closed", "Closed", "Cancelled", "Returned"]

        half = self.num_orders // 2

        # Oracle orders
        for _ in range(half):
            cust = random.choice(oracle_customers)
            num_lines = random.randint(1, 5)
            order_date = _random_date(2018, 2024)
            ship_dt = _random_date(2018, 2024) if random.random() < 0.8 else None
            status = random.choice(statuses_ora)
            lines_total = 0.0

            for _ in range(num_lines):
                prod = random.choice(oracle_products)
                qty = random.randint(1, 20)
                up = prod.UNIT_PRICE
                disc = round(random.uniform(0, 15), 2) if random.random() < 0.3 else None
                lt = round(qty * up * (1 - (disc or 0) / 100), 2)
                lines_total += lt
                ora_details.append(OracleOrderDetail(
                    ORDER_DTL_ID=ora_detail_id,
                    ORDER_ID=ora_order_id,
                    PROD_ID=prod.PROD_ID,
                    QTY=qty,
                    UNIT_PRICE=up,
                    LINE_TOTAL=lt,
                    DISCOUNT_PCT=disc,
                ))
                ora_detail_id += 1

            tax = round(lines_total * 0.08, 2)
            discount = round(lines_total * random.uniform(0, 0.05), 2)
            ora_orders.append(OracleOrder(
                ORDER_ID=ora_order_id,
                CUST_ID=cust.CUST_ID,
                ORDER_DT=order_date,
                SHIP_DT=ship_dt,
                ORDER_STATUS=status,
                ORDER_TOTAL=round(lines_total + tax - discount, 2),
                TAX_AMT=tax,
                DISCOUNT_AMT=discount if random.random() < 0.7 else None,
                SHIP_METHOD=random.choice(_SHIP_METHODS),
                CREATED_DT=order_date,
            ))
            ora_order_id += 1

        # SQL Server orders
        for _ in range(self.num_orders - half):
            cust = random.choice(sql_customers)
            num_lines = random.randint(1, 5)
            order_date = _random_date(2018, 2024)
            ship_dt = _random_date(2018, 2024) if random.random() < 0.8 else None
            status = random.choice(statuses_sql)
            lines_total = 0.0

            for _ in range(num_lines):
                prod = random.choice(sql_products)
                qty = random.randint(1, 20)
                up = prod.ListPrice
                disc = round(random.uniform(0, 15), 2) if random.random() < 0.3 else None
                lt = round(qty * up * (1 - (disc or 0) / 100), 2)
                lines_total += lt
                sql_details.append(SqlServerOrderDetail(
                    SalesOrderDetailID=sql_detail_id,
                    SalesOrderID=sql_order_id,
                    ProductID=prod.ProductID,
                    Quantity=qty,
                    UnitPrice=up,
                    LineTotal=lt,
                    DiscountPercent=disc,
                ))
                sql_detail_id += 1

            tax = round(lines_total * 0.08, 2)
            discount = round(lines_total * random.uniform(0, 0.05), 2)
            sql_orders.append(SqlServerOrder(
                SalesOrderID=sql_order_id,
                CustomerID=cust.CustomerID,
                OrderDate=order_date,
                ShipDate=ship_dt,
                OrderStatus=status,
                TotalAmount=round(lines_total + tax - discount, 2),
                TaxAmount=tax,
                DiscountAmount=discount if random.random() < 0.7 else None,
                ShippingMethod=random.choice(_SHIP_METHODS),
                CreatedDate=order_date,
            ))
            sql_order_id += 1

        return ora_orders, ora_details, sql_orders, sql_details

    # -- Convenience: generate everything at once -----------------------------

    def generate_all(self) -> Dict[str, Any]:
        """Generate a complete, correlated synthetic dataset.

        Returns a dict with keys for each table:
        ``oracle_customers``, ``sqlserver_customers``, ``oracle_products``, etc.
        """
        ora_custs, sql_custs = self.generate_customers()
        ora_prods, sql_prods = self.generate_products()
        ora_orders, ora_details, sql_orders, sql_details = self.generate_orders(
            ora_custs, sql_custs, ora_prods, sql_prods,
        )
        return {
            "oracle_customers": ora_custs,
            "sqlserver_customers": sql_custs,
            "oracle_products": ora_prods,
            "sqlserver_products": sql_prods,
            "oracle_orders": ora_orders,
            "oracle_order_details": ora_details,
            "sqlserver_orders": sql_orders,
            "sqlserver_order_details": sql_details,
            "shared_identities": self._shared_identities,
        }
