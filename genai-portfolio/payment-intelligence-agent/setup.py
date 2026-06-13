"""Payment Intelligence Agent - Setup Configuration."""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="payment-intelligence-agent",
    version="1.0.0",
    description=(
        "Snowflake Cortex Agent for payment analytics, "
        "PCI compliance RAG, and anomaly detection"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ravi Potluru",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit>=1.31.0",
        "snowflake-connector-python>=3.6.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.18.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "tiktoken>=0.5.0",
        "sqlparse>=0.4.4",
        "pyyaml>=6.0.1",
        "bleach>=6.1.0",
        "structlog>=23.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "ruff>=0.1.0",
            "mypy>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "payment-agent=src.app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial",
    ],
)
