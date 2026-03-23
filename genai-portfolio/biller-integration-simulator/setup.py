"""Package setup for biller-integration-simulator."""

from setuptools import setup, find_packages

setup(
    name="biller-integration-simulator",
    version="1.2.0",
    author="Ravi Potluru",
    author_email="ravi.potluru@example.com",
    description=(
        "Utility CIS-to-Payment Platform Integration Simulator — "
        "models Oracle CC&B schema mapping, payment processing, "
        "and three-way settlement reconciliation."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ravipotluru/biller-integration-simulator",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "config": ["*.yaml"],
    },
    python_requires=">=3.10",
    install_requires=[
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "biller-sim=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
