Data Science Portfolio
=====================

[![CI](https://github.com/raviteja0012/DataScience/actions/workflows/ci.yml/badge.svg)](https://github.com/raviteja0012/DataScience/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/raviteja0012/DataScience/branch/main/graph/badge.svg)](https://codecov.io/gh/raviteja0012/DataScience)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive collection of **36 production-ready data science projects** organized into 7 categories, covering machine learning, deep learning, data analysis, NLP, data engineering, MLOps, and GenAI/enterprise data systems. Each project is self-contained with synthetic data generation, making them immediately runnable.

Inspired by [50+ Data Project Ideas That Actually Get You Hired](https://penelopefitdatascientist.substack.com/p/50-data-project-ideas-that-actually) and reference implementations from the open-source data science community.

---

## GenAI & Enterprise Data Portfolio

Production-grade projects demonstrating enterprise data integration, AI-powered analytics, and payment systems engineering. Each project has full test suites (240 tests total), YAML-driven configuration, and runs entirely in demo mode with synthetic data.

| Project | Description | Key Tech | Tests |
|---------|-------------|----------|-------|
| [Biller Integration Simulator](genai-portfolio/biller-integration-simulator/) | Utility CIS-to-payment platform integration with Oracle CC&B schema models, YAML-driven onboarding, three-way settlement reconciliation | Python, dataclasses, PyYAML, structlog | 53 |
| [Payment Intelligence Agent](genai-portfolio/payment-intelligence-agent/) | Snowflake Cortex conversational agent with NL-to-SQL analytics, PCI-DSS v4.0 RAG, statistical anomaly detection, Streamlit chat UI | Streamlit, Plotly, FAISS, sentence-transformers | 101 |
| [Multi-Source Data Integration](genai-portfolio/multi-source-data-integration/) | M&A data migration pipeline with source discovery, schema mapping, identity resolution (5 algorithms), checkpoint/restart cutover | pandas, numpy, Jinja2, PyYAML | 86 |

See [genai-portfolio/README.md](genai-portfolio/README.md) for architecture details and quick start guides.

---

## Project Catalog

### 1. Machine Learning (8 projects)

| # | Project | Techniques | Key Skills |
|---|---------|-----------|------------|
| 01 | [House Price Prediction](projects/01-machine-learning/01-house-price-prediction/) | Linear/Ridge/Lasso/RF/GBM, GridSearchCV | Regression, Feature Engineering, Cross-Validation |
| 02 | [Customer Churn Classification](projects/01-machine-learning/02-customer-churn-classification/) | Logistic Regression/RF/XGBoost/SVM, SMOTE | Classification, Class Imbalance, Pipelines |
| 03 | [Customer Segmentation](projects/01-machine-learning/03-customer-segmentation-clustering/) | K-Means/DBSCAN/Hierarchical, PCA | Clustering, Dimensionality Reduction, Silhouette Analysis |
| 04 | [Sales Time Series Forecasting](projects/01-machine-learning/04-sales-time-series-forecasting/) | ARIMA, Exponential Smoothing, RF with Lag Features | Time Series, Walk-Forward Validation, Decomposition |
| 05 | [Credit Card Fraud Detection](projects/01-machine-learning/05-credit-card-fraud-detection/) | Isolation Forest/RF/GBM, SMOTE/Undersampling | Anomaly Detection, Extreme Imbalance, Threshold Tuning |
| 06 | [Medical Cost Prediction](projects/01-machine-learning/06-medical-cost-prediction/) | Ridge/Lasso/ElasticNet/RF/GBM | Regression, EDA, Residual Analysis, Feature Engineering |
| 07 | [Heart Disease Classification](projects/01-machine-learning/07-heart-disease-classification/) | LR/KNN/SVM/DT/RF/GBM, VotingClassifier, StackingClassifier | Ensemble Methods, Feature Selection, Clinical Threshold Optimization |
| 08 | [Loan Default Prediction](projects/01-machine-learning/08-loan-default-prediction/) | LR/RF/GBM, Probability Calibration | Risk Scoring, Scorecard Development, Profit Curves, Fairness Assessment |

### 2. Deep Learning (5 projects)

| # | Project | Architecture | Key Skills |
|---|---------|-------------|------------|
| 01 | [Image Classification CNN](projects/02-deep-learning/01-image-classification-cnn/) | CNN with Conv2D/MaxPooling/BatchNorm | CIFAR-10/MNIST, Data Augmentation, LR Scheduling |
| 02 | [Sentiment Analysis LSTM](projects/02-deep-learning/02-sentiment-analysis-lstm/) | Bi-LSTM with Attention Mechanism | IMDB Reviews, Text Preprocessing, Embedding |
| 03 | [Music Genre Classification](projects/02-deep-learning/03-music-genre-classification/) | CNN/DNN on Audio Features | MFCCs, Spectral Features, Multi-class Classification |
| 04 | [Face Mask Detection](projects/02-deep-learning/04-face-mask-detection/) | Transfer Learning (MobileNet/VGG-style) | Binary Classification, Data Augmentation, Callbacks |
| 05 | [Text Generation RNN](projects/02-deep-learning/05-text-generation-rnn/) | Character-level LSTM | Temperature Sampling, Teacher Forcing, Text Generation |

### 3. Data Analysis (5 projects)

| # | Project | Domain | Key Skills |
|---|---------|--------|------------|
| 01 | [Netflix EDA](projects/03-data-analysis/01-netflix-eda/) | Entertainment | Content Distribution, Temporal Trends, Geographic Analysis |
| 02 | [HR Analytics Dashboard](projects/03-data-analysis/02-hr-analytics-dashboard/) | Human Resources | Attrition Analysis, Statistical Tests (Chi-square, T-test) |
| 03 | [E-Commerce Analysis](projects/03-data-analysis/03-ecommerce-analysis/) | Retail | RFM Analysis, Cohort Analysis, Market Basket Analysis |
| 04 | [COVID-19 Analysis](projects/03-data-analysis/04-covid19-analysis/) | Public Health | CFR, Growth Rates, Moving Averages, Country Comparisons |
| 05 | [Salary Analysis](projects/03-data-analysis/05-salary-analysis/) | Compensation | Pay Gap Analysis, Regression, Confidence Intervals |

### 4. NLP (5 projects)

| # | Project | Techniques | Key Skills |
|---|---------|-----------|------------|
| 01 | [Spam Detection](projects/04-nlp/01-spam-detection/) | TF-IDF, Naive Bayes/LR/SVM/RF | Text Preprocessing, Sklearn Pipelines, Feature Engineering |
| 02 | [Topic Modeling](projects/04-nlp/02-topic-modeling/) | LDA, NMF | Coherence Optimization, Word Clouds, Document-Topic Assignment |
| 03 | [Text Summarization](projects/04-nlp/03-text-summarization/) | TF-IDF Scoring, TextRank | Extractive Summarization, Compression Ratios, ROUGE Metrics |
| 04 | [Named Entity Recognition](projects/04-nlp/04-named-entity-recognition/) | Rule-based NER with Regex/Gazetteers | Entity Extraction (PERSON/ORG/LOC/DATE/MONEY), Evaluation |
| 05 | [Movie Review Sentiment](projects/04-nlp/05-movie-review-sentiment/) | N-grams, Sentiment Lexicons | Aspect-Based Sentiment, Error Analysis, Word Importance |

### 5. Data Engineering (5 projects)

| # | Project | Architecture | Key Skills |
|---|---------|-------------|------------|
| 01 | [ETL Pipeline](projects/05-data-engineering/01-etl-pipeline/) | Extract-Transform-Load from CSV/JSON/API | Data Quality Checks, Logging, SQLite, Pipeline Reports |
| 02 | [Streaming Data Pipeline](projects/05-data-engineering/02-streaming-data-pipeline/) | Producer-Consumer with Threading | Windowed Aggregations, Anomaly Detection, Real-Time Stats |
| 03 | [Data Warehouse Design](projects/05-data-engineering/03-data-warehouse-design/) | Star Schema with Fact/Dimension Tables | OLAP Queries, SCD Type 2, SQLite, Data Lineage |
| 04 | [Web Scraping Pipeline](projects/05-data-engineering/04-web-scraping-pipeline/) | BeautifulSoup with Mock HTML | Rate Limiting, Retry Logic, Pagination, Ethics |
| 05 | [API Data Collector](projects/05-data-engineering/05-api-data-collector/) | REST API Framework with Mock Endpoints | Auth Patterns, Rate Limiting, Incremental Loading |

### 6. MLOps & Deployment (5 projects)

| # | Project | Technology | Key Skills |
|---|---------|-----------|------------|
| 01 | [Model Serving Flask](projects/06-mlops-deployment/01-model-serving-flask/) | Flask REST API | /predict, /health, /model-info Endpoints, Input Validation |
| 02 | [ML Pipeline Automation](projects/06-mlops-deployment/02-ml-pipeline-automation/) | Automated Pipeline Orchestration | Experiment Tracking, Model Registry, Reproducibility |
| 03 | [Model Monitoring Dashboard](projects/06-mlops-deployment/03-model-monitoring-dashboard/) | Drift Detection System | KS Test, PSI, Concept Drift, Retraining Triggers |
| 04 | [Docker ML Deployment](projects/06-mlops-deployment/04-docker-ml-deployment/) | Docker + FastAPI | Containerization, Multi-stage Build, docker-compose |
| 05 | [CI/CD ML Pipeline](projects/06-mlops-deployment/05-ci-cd-ml-pipeline/) | CI/CD Simulation + GitHub Actions | A/B Testing, Canary Deployment, Rollback Logic |

---

## Getting Started

```bash
# Clone the repo
git clone <this-repo-url>
cd DataScience

# Run any classic project (each is self-contained)
cd projects/01-machine-learning/01-house-price-prediction
pip install -r requirements.txt
python house_price_prediction.py

# Run any GenAI portfolio project
cd genai-portfolio/payment-intelligence-agent
pip install -r requirements.txt
python -m pytest tests/ -v          # run tests
python -m src.app                   # launch Streamlit app (demo mode)
```

Each project generates its own synthetic data, so no external datasets are needed.

### Running All Tests

```bash
# GenAI portfolio tests (240 total)
for proj in biller-integration-simulator payment-intelligence-agent multi-source-data-integration; do
  python -m pytest genai-portfolio/$proj/tests/ -v
done
```

## Production-Ready Framework

Following the six-component framework for production-ready data science projects:

1. **Clear Problem Definition** - Each project starts with a well-defined business problem
2. **Data Pipeline Design** - Real data processing pipelines, not just static CSV reads
3. **Reproducible Code** - Random seeds, requirements.txt, self-contained scripts
4. **Testing & Monitoring** - Model evaluation metrics, drift detection, quality checks
5. **Deployment Strategy** - Flask/FastAPI serving, Docker containerization, CI/CD
6. **Communication & Documentation** - Clear output, visualizations, and insights

---

## DDIA References

This repository also includes literature references for [Designing Data-Intensive Applications](http://dataintensive.net/) by [Martin Kleppmann](http://martin.kleppmann.com/).

### Chapters

1.  [References for Chapter  1](chapter-01-refs.md)
2.  [References for Chapter  2](chapter-02-refs.md)
3.  [References for Chapter  3](chapter-03-refs.md)
4.  [References for Chapter  4](chapter-04-refs.md)
5.  [References for Chapter  5](chapter-05-refs.md)
6.  [References for Chapter  6](chapter-06-refs.md)
7.  [References for Chapter  7](chapter-07-refs.md)
8.  [References for Chapter  8](chapter-08-refs.md)
9.  [References for Chapter  9](chapter-09-refs.md)
10. [References for Chapter 10](chapter-10-refs.md)
11. [References for Chapter 11](chapter-11-refs.md)
12. [References for Chapter 12](chapter-12-refs.md)

### Maps

* [Poster of maps in PDF format](ddia-poster.pdf)
* [Poster of maps in JPEG format](ddia-poster.jpg)
