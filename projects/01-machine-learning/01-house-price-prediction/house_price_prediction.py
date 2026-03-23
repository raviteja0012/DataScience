"""
House Price Prediction with Multiple Regression Models
=======================================================

This project demonstrates a complete machine learning pipeline for predicting
house prices using the California Housing dataset from scikit-learn.

Pipeline stages:
    1. Data loading and exploratory analysis
    2. Feature engineering (polynomial features, interaction terms)
    3. Preprocessing (scaling, handling skewness)
    4. Model training with multiple algorithms
    5. Hyperparameter tuning via GridSearchCV
    6. Cross-validation and model comparison
    7. Feature importance visualization
    8. Saving the best model with joblib

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
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    KFold,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

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
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
sns.set_style("whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})


# ---------------------------------------------------------------------------
# 1. Data Loading & Exploration
# ---------------------------------------------------------------------------
def load_and_explore_data():
    """Load the California Housing dataset and return a DataFrame with EDA plots.

    Returns
    -------
    df : pd.DataFrame
        Full dataset with feature columns and the target ``MedHouseVal``.
    """
    logger.info("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # includes target column 'MedHouseVal'

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Features: {list(housing.feature_names)}")
    logger.info(f"Target: MedHouseVal (median house value in $100k)")

    # --- Summary statistics ---
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(df.describe().round(3).to_string())

    # --- Missing values ---
    missing = df.isnull().sum()
    print(f"\nMissing values:\n{missing[missing > 0] if missing.any() else 'None'}")

    # --- Distribution of target variable ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["MedHouseVal"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].set_title("Distribution of Median House Value", fontsize=13)
    axes[0].set_xlabel("Median House Value ($100k)")
    axes[0].set_ylabel("Frequency")

    # Correlation heatmap
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=axes[1], square=True, linewidths=0.5,
    )
    axes[1].set_title("Feature Correlation Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_eda_overview.png"), dpi=150)
    plt.close()
    logger.info("Saved EDA overview plot.")

    # --- Pairwise scatter for top-correlated features ---
    top_features = corr["MedHouseVal"].drop("MedHouseVal").abs().nlargest(4).index.tolist()
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, feat in zip(axes, top_features):
        ax.scatter(df[feat], df["MedHouseVal"], alpha=0.15, s=5, color="steelblue")
        ax.set_xlabel(feat)
        ax.set_ylabel("MedHouseVal")
        ax.set_title(f"{feat} vs Target")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_scatter_top_features.png"), dpi=150)
    plt.close()
    logger.info("Saved scatter plots for top-correlated features.")

    return df


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df):
    """Create new features from the raw California Housing data.

    New features include:
        - Rooms per household
        - Bedrooms ratio (bedrooms / total rooms)
        - Population per household
        - Selected polynomial / interaction terms

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df_eng : pd.DataFrame
        DataFrame with original + engineered features.
    """
    logger.info("Engineering features...")
    df_eng = df.copy()

    # Domain-driven features
    df_eng["RoomsPerHousehold"] = df_eng["AveRooms"] * df_eng["AveOccup"]
    df_eng["BedroomRatio"] = df_eng["AveBedrms"] / (df_eng["AveRooms"] + 1e-8)
    df_eng["PopPerHousehold"] = df_eng["Population"] / (df_eng["HouseAge"] + 1)

    # Interaction: income * location proxy
    df_eng["IncomeLatitude"] = df_eng["MedInc"] * df_eng["Latitude"]
    df_eng["IncomeLongitude"] = df_eng["MedInc"] * df_eng["Longitude"]

    # Polynomial features for the strongest predictor (MedInc)
    df_eng["MedInc_sq"] = df_eng["MedInc"] ** 2
    df_eng["MedInc_log"] = np.log1p(df_eng["MedInc"])

    logger.info(f"Feature count after engineering: {df_eng.shape[1] - 1}")
    return df_eng


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
def preprocess_data(df, target_col="MedHouseVal"):
    """Split and scale the data.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
    feature_names : list[str]
    scaler : StandardScaler (fitted)
    """
    logger.info("Preprocessing data...")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, feature_names, scaler


# ---------------------------------------------------------------------------
# 4. Model Definitions
# ---------------------------------------------------------------------------
def get_models():
    """Return a dictionary of model name -> (estimator, param_grid) pairs.

    The param grids are intentionally small so the script finishes in a
    reasonable time while still demonstrating GridSearchCV.
    """
    models = {
        "Linear Regression": (
            LinearRegression(),
            {},  # no hyperparams to tune
        ),
        "Ridge": (
            Ridge(random_state=RANDOM_STATE),
            {"alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "Lasso": (
            Lasso(random_state=RANDOM_STATE, max_iter=10000),
            {"alpha": [0.001, 0.01, 0.1, 1.0]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            },
        ),
    }
    return models


# ---------------------------------------------------------------------------
# 5. Training, Tuning & Evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """Train all models, perform hyperparameter tuning, and evaluate.

    Returns
    -------
    results : dict
        model_name -> {best_estimator, cv_rmse, test_rmse, test_mae, test_r2}
    """
    models = get_models()
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print("\n" + "=" * 70)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 70)

    for name, (estimator, param_grid) in models.items():
        logger.info(f"Training {name}...")

        # --- GridSearchCV (or simple fit if no params) ---
        if param_grid:
            grid = GridSearchCV(
                estimator,
                param_grid,
                cv=kf,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                verbose=0,
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
        else:
            best_model = estimator
            best_model.fit(X_train, y_train)
            best_params = {}

        # --- Cross-validation on training set ---
        cv_scores = cross_val_score(
            best_model, X_train, y_train, cv=kf,
            scoring="neg_root_mean_squared_error",
        )
        cv_rmse = -cv_scores.mean()
        cv_rmse_std = cv_scores.std()

        # --- Test set evaluation ---
        y_pred = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)

        results[name] = {
            "best_estimator": best_model,
            "best_params": best_params,
            "cv_rmse": cv_rmse,
            "cv_rmse_std": cv_rmse_std,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "y_pred": y_pred,
        }

        print(f"\n--- {name} ---")
        if best_params:
            print(f"  Best params   : {best_params}")
        print(f"  CV RMSE       : {cv_rmse:.4f} (+/- {cv_rmse_std:.4f})")
        print(f"  Test RMSE     : {test_rmse:.4f}")
        print(f"  Test MAE      : {test_mae:.4f}")
        print(f"  Test R2       : {test_r2:.4f}")

    return results


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------
def plot_model_comparison(results):
    """Create bar charts comparing models on RMSE, MAE, and R2."""
    names = list(results.keys())
    rmse_vals = [results[n]["test_rmse"] for n in names]
    mae_vals = [results[n]["test_mae"] for n in names]
    r2_vals = [results[n]["test_r2"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # RMSE comparison
    bars = axes[0].barh(names, rmse_vals, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("RMSE")
    axes[0].set_title("Model Comparison - RMSE (lower is better)")
    for bar, val in zip(bars, rmse_vals):
        axes[0].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)

    # MAE comparison
    bars = axes[1].barh(names, mae_vals, color="coral", edgecolor="black")
    axes[1].set_xlabel("MAE")
    axes[1].set_title("Model Comparison - MAE (lower is better)")
    for bar, val in zip(bars, mae_vals):
        axes[1].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)

    # R2 comparison
    bars = axes[2].barh(names, r2_vals, color="seagreen", edgecolor="black")
    axes[2].set_xlabel("R-squared")
    axes[2].set_title("Model Comparison - R² (higher is better)")
    for bar, val in zip(bars, r2_vals):
        axes[2].text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_model_comparison.png"), dpi=150)
    plt.close()
    logger.info("Saved model comparison chart.")


def plot_feature_importance(results, feature_names):
    """Plot feature importance for tree-based models and coefficients for linear."""
    tree_models = ["Random Forest", "Gradient Boosting"]
    linear_models = ["Ridge", "Lasso", "Linear Regression"]

    # --- Tree-based feature importance ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, name in zip(axes, tree_models):
        model = results[name]["best_estimator"]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # top 15

        ax.barh(
            [feature_names[i] for i in indices][::-1],
            importances[indices][::-1],
            color="steelblue", edgecolor="black",
        )
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"{name} - Top 15 Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_feature_importance_trees.png"), dpi=150)
    plt.close()

    # --- Linear model coefficients ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    for ax, name in zip(axes, linear_models):
        model = results[name]["best_estimator"]
        coefs = model.coef_
        indices = np.argsort(np.abs(coefs))[::-1][:15]

        colors = ["coral" if coefs[i] < 0 else "steelblue" for i in indices]
        ax.barh(
            [feature_names[i] for i in indices][::-1],
            coefs[indices][::-1],
            color=colors[::-1], edgecolor="black",
        )
        ax.set_xlabel("Coefficient Value")
        ax.set_title(f"{name} - Top 15 Coefficients")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_feature_importance_linear.png"), dpi=150)
    plt.close()
    logger.info("Saved feature importance plots.")


def plot_predictions_vs_actual(results, y_test):
    """Scatter plot of predicted vs actual for each model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, info) in zip(axes, results.items()):
        y_pred = info["y_pred"]
        ax.scatter(y_test, y_pred, alpha=0.2, s=8, color="steelblue")
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5, label="Perfect")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}\nR²={info['test_r2']:.4f}")
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_predicted_vs_actual.png"), dpi=150)
    plt.close()
    logger.info("Saved predicted-vs-actual plots.")


def plot_residuals(results, y_test):
    """Residual distribution for the best model."""
    best_name = min(results, key=lambda n: results[n]["test_rmse"])
    y_pred = results[best_name]["y_pred"]
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual scatter
    axes[0].scatter(y_pred, residuals, alpha=0.2, s=8, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_xlabel("Predicted Value")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals vs Predicted ({best_name})")

    # Residual histogram
    axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Residual Distribution ({best_name})")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_residuals.png"), dpi=150)
    plt.close()
    logger.info("Saved residual plots.")


# ---------------------------------------------------------------------------
# 7. Save Best Model
# ---------------------------------------------------------------------------
def save_best_model(results, scaler, feature_names):
    """Persist the best model, scaler, and feature names using joblib.

    The best model is selected by the lowest test RMSE.
    """
    best_name = min(results, key=lambda n: results[n]["test_rmse"])
    best_info = results[best_name]

    artifact = {
        "model_name": best_name,
        "model": best_info["best_estimator"],
        "scaler": scaler,
        "feature_names": feature_names,
        "metrics": {
            "test_rmse": best_info["test_rmse"],
            "test_mae": best_info["test_mae"],
            "test_r2": best_info["test_r2"],
        },
    }

    model_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
    joblib.dump(artifact, model_path)

    print(f"\n{'=' * 70}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'=' * 70}")
    print(f"  Test RMSE : {best_info['test_rmse']:.4f}")
    print(f"  Test MAE  : {best_info['test_mae']:.4f}")
    print(f"  Test R2   : {best_info['test_r2']:.4f}")
    print(f"  Saved to  : {model_path}")

    return best_name


# ---------------------------------------------------------------------------
# 8. Summary Table
# ---------------------------------------------------------------------------
def print_summary_table(results):
    """Print a formatted comparison table of all models."""
    print(f"\n{'=' * 70}")
    print("FINAL MODEL COMPARISON")
    print(f"{'=' * 70}")
    header = f"{'Model':<25} {'CV RMSE':>10} {'Test RMSE':>10} {'Test MAE':>10} {'Test R2':>10}"
    print(header)
    print("-" * len(header))

    for name, info in sorted(results.items(), key=lambda x: x[1]["test_rmse"]):
        print(
            f"{name:<25} "
            f"{info['cv_rmse']:>10.4f} "
            f"{info['test_rmse']:>10.4f} "
            f"{info['test_mae']:>10.4f} "
            f"{info['test_r2']:>10.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Orchestrate the full house-price-prediction pipeline."""
    print("\n" + "#" * 70)
    print("#  HOUSE PRICE PREDICTION - CALIFORNIA HOUSING DATASET")
    print("#" * 70)

    # Step 1 - Load & explore
    df = load_and_explore_data()

    # Step 2 - Feature engineering
    df = engineer_features(df)

    # Step 3 - Preprocess
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(df)

    # Step 4 & 5 - Train, tune, evaluate
    results = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)

    # Step 6 - Visualize
    plot_model_comparison(results)
    plot_feature_importance(results, feature_names)
    plot_predictions_vs_actual(results, y_test)
    plot_residuals(results, y_test)

    # Step 7 - Save best model
    save_best_model(results, scaler, feature_names)

    # Step 8 - Summary
    print_summary_table(results)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
