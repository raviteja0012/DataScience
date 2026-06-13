"""Package setup for the Multi-Source Data Integration Framework."""

from pathlib import Path
from setuptools import setup, find_packages

_ROOT = Path(__file__).parent

setup(
    name="multi-source-data-integration",
    version="2.1.0",
    description=(
        "Configurable pipeline for acquired company data integration: "
        "source discovery, schema mapping, identity resolution, "
        "cutover management, and reconciliation validation."
    ),
    long_description=(_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Ravi Potluru",
    author_email="rpotluru@kent.edu",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    python_requires=">=3.9",
    install_requires=[
        "PyYAML>=6.0",
    ],
    extras_require={
        "databases": [
            "sqlalchemy>=2.0",
            "cx_Oracle>=8.0",
            "pyodbc>=4.0",
            "snowflake-connector-python>=3.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "mypy>=1.0",
            "ruff>=0.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "msdi=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
