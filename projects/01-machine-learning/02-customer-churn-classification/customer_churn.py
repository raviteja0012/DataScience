"""
Customer Churn Classification for Telecom
==========================================

This project builds a complete classification pipeline to predict customer
churn in a synthetic telecom dataset.  It covers:

    1. Realistic synthetic data generation (tenure, charges, contract, etc.)
    2. Exploratory data analysis
    3. Preprocessing pipelines (StandardScaler, OneHotEncoder)
    4. Handling class imbalance with SMOTE
    5. Multiple classifiers: Logistic Regression, Random Forest, XGBoost, SVM
    6. Stratified K-Fold cross-validation
    7. ROC curves, precision-recall curves, confusion matrices
    8. Feature importance analysis

Author: Data Science Portfolio
Date: 2026-02-21
"""

import os
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# XGBoost -------------------------------------------------------------------
try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.warning(
        "XGBoost not installed. The XGBoost model will be skipped. "
        "Install with: pip install xgboost"
    )

# SMOTE ---------------------------------------------------------------------
try:
    from imblearn.over_sampling import SMOTE

    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    logging.warning(
        "imbalanced-learn not installed. SMOTE will be skipped. "
        "Install with: pip install imbalanced-learn"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_SAMPLES = 7000  # synthetic dataset size
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
sns.set_style("whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})


# ---------------------------------------------------------------------------
# 1. Synthetic Data Generation
# ---------------------------------------------------------------------------
def generate_churn_data(n_samples=N_SAMPLES):
    """Generate a realistic synthetic telecom customer churn dataset.

    Features
    --------
    - gender : Male / Female
    - SeniorCitizen : 0 / 1
    - Partner : Yes / No
    - Dependents : Yes / No
    - tenure : months as customer (0-72)
    - PhoneService : Yes / No
    - InternetService : DSL / Fiber optic / No
    - Contract : Month-to-month / One year / Two year
    - PaperlessBilling : Yes / No
    - PaymentMethod : Electronic check / Mailed check / Bank transfer / Credit card
    - MonthlyCharges : continuous
    - TotalCharges : continuous (correlated with tenure * MonthlyCharges)

    Target: Churn (0 / 1)  -- ~26% churn rate

    Returns
    -------
    df : pd.DataFrame
    """
    logger.info(f"Generating synthetic churn data ({n_samples} samples)...")
    rng = np.random.RandomState(RANDOM_STATE)

    gender = rng.choice(["Male", "Female"], n_samples)
    senior = rng.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n_samples, p=[0.30, 0.70])
    tenure = rng.exponential(scale=25, size=n_samples).clip(0, 72).astype(int)
    phone_service = rng.choice(["Yes", "No"], n_samples, p=[0.90, 0.10])
    internet = rng.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]
    )
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.21, 0.24]
    )
    paperless = rng.choice(["Yes", "No"], n_samples, p=[0.60, 0.40])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_samples,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # Monthly charges depend on internet service
    base_charge = np.where(
        internet == "Fiber optic",
        rng.normal(80, 15, n_samples),
        np.where(internet == "DSL", rng.normal(55, 12, n_samples), rng.normal(25, 8, n_samples)),
    ).clip(18, 120)
    monthly_charges = base_charge.round(2)
    total_charges = (tenure * monthly_charges + rng.normal(0, 50, n_samples)).clip(0).round(2)

    # --- Churn probability (logistic function of risk factors) ---
    risk = (
        -1.2
        + 0.8 * (contract == "Month-to-month").astype(float)
        - 0.6 * (contract == "Two year").astype(float)
        + 0.5 * (internet == "Fiber optic").astype(float)
        + 0.4 * (payment == "Electronic check").astype(float)
        - 0.03 * tenure
        + 0.01 * monthly_charges
        + 0.3 * senior
        - 0.2 * (partner == "Yes").astype(float)
        + rng.normal(0, 0.5, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-risk))
    churn = (rng.random(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "InternetService": internet,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )

    logger.info(f"Churn rate: {churn.mean():.2%}")
    return df


# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
def exploratory_analysis(df):
    """Generate exploratory plots and summary statistics."""
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(df.describe(include="all").to_string())
    print(f"\nChurn distribution:\n{df['Churn'].value_counts(normalize=True).to_string()}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Churn by contract type
    ct = pd.crosstab(df["Contract"], df["Churn"], normalize="index")
    ct.plot(kind="bar", stacked=True, ax=axes[0, 0], color=["steelblue", "coral"])
    axes[0, 0].set_title("Churn Rate by Contract Type")
    axes[0, 0].set_ylabel("Proportion")
    axes[0, 0].legend(["No Churn", "Churn"])

    # Churn by internet service
    ct2 = pd.crosstab(df["InternetService"], df["Churn"], normalize="index")
    ct2.plot(kind="bar", stacked=True, ax=axes[0, 1], color=["steelblue", "coral"])
    axes[0, 1].set_title("Churn Rate by Internet Service")
    axes[0, 1].set_ylabel("Proportion")

    # Monthly charges distribution by churn
    for label, color in [(0, "steelblue"), (1, "coral")]:
        subset = df[df["Churn"] == label]["MonthlyCharges"]
        axes[0, 2].hist(subset, bins=30, alpha=0.6, label=f"Churn={label}", color=color, edgecolor="black")
    axes[0, 2].set_title("Monthly Charges by Churn")
    axes[0, 2].legend()

    # Tenure distribution by churn
    for label, color in [(0, "steelblue"), (1, "coral")]:
        subset = df[df["Churn"] == label]["tenure"]
        axes[1, 0].hist(subset, bins=30, alpha=0.6, label=f"Churn={label}", color=color, edgecolor="black")
    axes[1, 0].set_title("Tenure by Churn")
    axes[1, 0].legend()

    # Churn by payment method
    ct3 = pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index")
    ct3.plot(kind="bar", stacked=True, ax=axes[1, 1], color=["steelblue", "coral"])
    axes[1, 1].set_title("Churn by Payment Method")
    axes[1, 1].tick_params(axis="x", rotation=30)

    # Overall churn pie
    df["Churn"].value_counts().plot.pie(
        ax=axes[1, 2], autopct="%1.1f%%", colors=["steelblue", "coral"],
        labels=["No Churn", "Churn"], startangle=90,
    )
    axes[1, 2].set_ylabel("")
    axes[1, 2].set_title("Overall Churn Rate")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_eda_churn.png"), dpi=150)
    plt.close()
    logger.info("Saved EDA plots.")


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df):
    """Build preprocessing pipelines and split data.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
    preprocessor : fitted ColumnTransformer
    feature_names : list[str]
    """
    logger.info("Preprocessing data...")

    target = "Churn"
    X = df.drop(columns=[target])
    y = df[target]

    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    logger.info(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    logger.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    # Fit preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Recover feature names
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
    feature_names = numeric_cols + cat_feature_names

    logger.info(f"Processed feature count: {len(feature_names)}")

    # --- SMOTE for class imbalance ---
    if HAS_SMOTE:
        logger.info("Applying SMOTE to balance training data...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
        logger.info(f"After SMOTE - class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    else:
        logger.info("SMOTE unavailable; proceeding with imbalanced data.")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names


# ---------------------------------------------------------------------------
# 4. Model Definitions
# ---------------------------------------------------------------------------
def get_models():
    """Return dict of model_name -> (estimator, param_grid)."""
    models = {
        "Logistic Regression": (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver="saga"),
            {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, 20],
                "min_samples_split": [2, 5],
            },
        ),
        "SVM": (
            SVC(random_state=RANDOM_STATE, probability=True),
            {"C": [0.1, 1, 10], "kernel": ["rbf"], "gamma": ["scale", "auto"]},
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
            },
        )

    return models


# ---------------------------------------------------------------------------
# 5. Training & Evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """Train models with GridSearchCV and evaluate on the test set.

    Returns
    -------
    results : dict  (model_name -> metrics and fitted model)
    """
    models = get_models()
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print("\n" + "=" * 70)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 70)

    for name, (estimator, param_grid) in models.items():
        logger.info(f"Training {name}...")

        grid = GridSearchCV(
            estimator,
            param_grid,
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # Cross-val AUC
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=skf, scoring="roc_auc")

        # Test predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "best_estimator": best_model,
            "best_params": grid.best_params_,
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
        results[name] = metrics

        print(f"\n--- {name} ---")
        print(f"  Best params : {grid.best_params_}")
        print(f"  CV AUC      : {metrics['cv_auc_mean']:.4f} (+/- {metrics['cv_auc_std']:.4f})")
        print(f"  Accuracy    : {metrics['accuracy']:.4f}")
        print(f"  Precision   : {metrics['precision']:.4f}")
        print(f"  Recall      : {metrics['recall']:.4f}")
        print(f"  F1 Score    : {metrics['f1']:.4f}")
        print(f"  ROC AUC     : {metrics['roc_auc']:.4f}")

    return results


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------
def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models on one figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "coral", "seagreen", "darkorange", "purple"]

    for (name, info), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, info["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={info['roc_auc']:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Model Comparison", fontsize=14)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_roc_curves.png"), dpi=150)
    plt.close()
    logger.info("Saved ROC curves.")


def plot_precision_recall_curves(results, y_test):
    """Plot precision-recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "coral", "seagreen", "darkorange", "purple"]

    for (name, info), color in zip(results.items(), colors):
        precision, recall, _ = precision_recall_curve(y_test, info["y_prob"])
        ap = average_precision_score(y_test, info["y_prob"])
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=color, lw=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_precision_recall_curves.png"), dpi=150)
    plt.close()
    logger.info("Saved precision-recall curves.")


def plot_confusion_matrices(results, y_test):
    """Plot confusion matrices for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, info) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, info["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}\nF1={info['f1']:.3f}")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_confusion_matrices.png"), dpi=150)
    plt.close()
    logger.info("Saved confusion matrices.")


def plot_feature_importance(results, feature_names):
    """Plot feature importances for tree-based models and coefficients for LR."""
    importances_dict = {}

    for name, info in results.items():
        model = info["best_estimator"]
        if hasattr(model, "feature_importances_"):
            importances_dict[name] = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances_dict[name] = np.abs(model.coef_[0])

    if not importances_dict:
        return

    n = len(importances_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, importances) in zip(axes, importances_dict.items()):
        indices = np.argsort(importances)[::-1][:15]
        names_sorted = [feature_names[i] for i in indices][::-1]
        vals_sorted = importances[indices][::-1]

        ax.barh(names_sorted, vals_sorted, color="steelblue", edgecolor="black")
        ax.set_xlabel("Importance")
        ax.set_title(f"{name} - Top 15 Features")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_feature_importance.png"), dpi=150)
    plt.close()
    logger.info("Saved feature importance plots.")


def plot_model_comparison(results):
    """Bar chart comparing all models across key metrics."""
    names = list(results.keys())
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.15

    for i, (metric, label) in enumerate(zip(metrics_to_plot, labels)):
        vals = [results[n][metric] for n in names]
        ax.bar(x + i * width, vals, width, label=label, edgecolor="black")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison Across Metrics")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_model_comparison.png"), dpi=150)
    plt.close()
    logger.info("Saved model comparison chart.")


# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
def print_summary(results):
    """Print a final summary table."""
    print(f"\n{'=' * 80}")
    print("FINAL MODEL COMPARISON")
    print(f"{'=' * 80}")
    header = (
        f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
        f"{'F1':>8} {'ROC AUC':>9} {'CV AUC':>9}"
    )
    print(header)
    print("-" * len(header))

    for name, info in sorted(results.items(), key=lambda x: -x[1]["roc_auc"]):
        print(
            f"{name:<22} "
            f"{info['accuracy']:>9.4f} "
            f"{info['precision']:>10.4f} "
            f"{info['recall']:>8.4f} "
            f"{info['f1']:>8.4f} "
            f"{info['roc_auc']:>9.4f} "
            f"{info['cv_auc_mean']:>9.4f}"
        )

    # Classification report for best model
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"\nBest model by ROC AUC: {best_name}")
    print("\nDetailed Classification Report:")
    # We need y_test for this; handled in main


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Orchestrate the full churn-prediction pipeline."""
    print("\n" + "#" * 70)
    print("#  CUSTOMER CHURN CLASSIFICATION - TELECOM DATASET")
    print("#" * 70)

    # Step 1 - Generate data
    df = generate_churn_data()

    # Step 2 - EDA
    exploratory_analysis(df)

    # Step 3 - Preprocess
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess(df)

    # Step 4 & 5 - Train and evaluate
    results = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)

    # Step 6 - Visualizations
    plot_roc_curves(results, y_test)
    plot_precision_recall_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_feature_importance(results, feature_names)
    plot_model_comparison(results)

    # Step 7 - Summary
    print_summary(results)

    # Detailed report for best model
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"\n--- Classification Report: {best_name} ---")
    print(classification_report(y_test, results[best_name]["y_pred"],
                                target_names=["No Churn", "Churn"]))

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
