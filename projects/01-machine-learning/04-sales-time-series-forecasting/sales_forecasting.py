"""
Sales Time Series Forecasting
==============================

This project demonstrates time series forecasting on synthetic daily sales
data using multiple approaches:

    1. Synthetic data generation (trend, seasonality, holidays, noise)
    2. Time series decomposition (trend, seasonal, residual)
    3. Models:
       a. ARIMA (via statsmodels)
       b. Exponential Smoothing (Holt-Winters)
       c. Prophet-style manual implementation (Fourier terms + trend)
       d. Random Forest with lag features
    4. Walk-forward (expanding window) validation
    5. Forecast visualization with confidence intervals
    6. Metrics: MAPE, RMSE, MAE

Author: Data Science Portfolio
Date: 2026-02-21
"""

import os
import warnings
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# statsmodels
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logging.warning(
        "statsmodels not installed. ARIMA and Exponential Smoothing will be skipped. "
        "Install with: pip install statsmodels"
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
FORECAST_HORIZON = 30  # days to forecast
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
sns.set_style("whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})


# ===========================================================================
# 1. Synthetic Data Generation
# ===========================================================================
def generate_sales_data(start_date="2021-01-01", periods=1095):
    """Generate realistic synthetic daily sales data.

    Components
    ----------
    - Linear trend with slight acceleration
    - Weekly seasonality (weekends lower)
    - Annual seasonality (peak in Nov-Dec, dip in Jan-Feb)
    - Holiday spikes (Black Friday, Christmas, New Year)
    - Random noise

    Parameters
    ----------
    start_date : str
    periods : int  (days, default ~3 years)

    Returns
    -------
    df : pd.DataFrame with columns ['date', 'sales']
    """
    logger.info(f"Generating {periods} days of synthetic sales data...")
    rng = np.random.RandomState(RANDOM_STATE)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    t = np.arange(periods)

    # --- Trend ---
    trend = 500 + 0.3 * t + 0.0001 * t ** 2

    # --- Weekly seasonality (day-of-week effect) ---
    dow = dates.dayofweek  # 0=Mon ... 6=Sun
    weekly = np.where(dow == 5, -40, np.where(dow == 6, -70, 20))  # lower weekends

    # --- Annual seasonality (Fourier approximation) ---
    day_of_year = dates.dayofyear
    annual = (
        60 * np.sin(2 * np.pi * day_of_year / 365.25)
        + 40 * np.cos(2 * np.pi * day_of_year / 365.25)
        + 30 * np.sin(4 * np.pi * day_of_year / 365.25)
    )

    # --- Holiday effects ---
    holiday_boost = np.zeros(periods)
    for i, d in enumerate(dates):
        # Black Friday region (last week of November)
        if d.month == 11 and d.day >= 24 and d.day <= 30:
            holiday_boost[i] = 250
        # Christmas build-up (Dec 15-25)
        elif d.month == 12 and d.day >= 15 and d.day <= 25:
            holiday_boost[i] = 200 + 20 * (d.day - 15)
        # New Year dip
        elif d.month == 1 and d.day <= 5:
            holiday_boost[i] = -100
        # Summer sale (July 4 week)
        elif d.month == 7 and d.day >= 1 and d.day <= 7:
            holiday_boost[i] = 100

    # --- Noise ---
    noise = rng.normal(0, 40, periods)

    # --- Combine ---
    sales = trend + weekly + annual + holiday_boost + noise
    sales = np.maximum(sales, 50)  # sales cannot be below 50

    df = pd.DataFrame({"date": dates, "sales": sales.round(2)})
    df = df.set_index("date")

    logger.info(f"Sales range: [{df['sales'].min():.0f}, {df['sales'].max():.0f}]")
    return df


# ===========================================================================
# 2. EDA & Decomposition
# ===========================================================================
def exploratory_analysis(df):
    """Generate time series EDA plots."""
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(df.describe().round(2).to_string())

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)

    # Full time series
    axes[0].plot(df.index, df["sales"], color="steelblue", lw=0.8)
    axes[0].set_title("Daily Sales Over Time", fontsize=13)
    axes[0].set_ylabel("Sales ($)")

    # Monthly aggregation
    monthly = df.resample("ME").mean()
    axes[1].plot(monthly.index, monthly["sales"], "o-", color="coral", lw=1.5)
    axes[1].set_title("Monthly Average Sales", fontsize=13)
    axes[1].set_ylabel("Avg Sales ($)")

    # Distribution
    axes[2].hist(df["sales"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[2].set_title("Sales Distribution", fontsize=13)
    axes[2].set_xlabel("Sales ($)")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_eda_timeseries.png"), dpi=150)
    plt.close()

    # Weekly pattern
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_avg = df.groupby(df.index.dayofweek)["sales"].mean()
    axes[0].bar(day_names, dow_avg.values, color="steelblue", edgecolor="black")
    axes[0].set_title("Average Sales by Day of Week")
    axes[0].set_ylabel("Avg Sales ($)")

    # Monthly pattern
    month_avg = df.groupby(df.index.month)["sales"].mean()
    axes[1].bar(range(1, 13), month_avg.values, color="coral", edgecolor="black")
    axes[1].set_title("Average Sales by Month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Avg Sales ($)")
    axes[1].set_xticks(range(1, 13))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_seasonality_patterns.png"), dpi=150)
    plt.close()

    logger.info("Saved EDA plots.")


def decompose_series(df):
    """Perform additive seasonal decomposition and plot."""
    if not HAS_STATSMODELS:
        logger.warning("statsmodels unavailable; skipping decomposition.")
        return

    logger.info("Decomposing time series (period=365)...")
    result = seasonal_decompose(df["sales"], model="additive", period=365)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    result.observed.plot(ax=axes[0], color="steelblue", lw=0.7)
    axes[0].set_title("Observed")
    result.trend.plot(ax=axes[1], color="coral", lw=1.5)
    axes[1].set_title("Trend")
    result.seasonal.plot(ax=axes[2], color="seagreen", lw=0.7)
    axes[2].set_title("Seasonal")
    result.resid.plot(ax=axes[3], color="gray", lw=0.5)
    axes[3].set_title("Residual")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_decomposition.png"), dpi=150)
    plt.close()
    logger.info("Saved decomposition plot.")

    # Stationarity test
    adf_result = adfuller(df["sales"].dropna())
    print(f"\nAugmented Dickey-Fuller Test:")
    print(f"  ADF Statistic : {adf_result[0]:.4f}")
    print(f"  p-value       : {adf_result[1]:.4f}")
    print(f"  Stationary?   : {'Yes' if adf_result[1] < 0.05 else 'No (differencing needed)'}")


# ===========================================================================
# 3. Metrics
# ===========================================================================
def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_forecast(y_true, y_pred, model_name="Model"):
    """Compute and print RMSE, MAE, MAPE.

    Returns
    -------
    dict with metric values
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)

    print(f"  {model_name:<30} RMSE={rmse:>8.2f}  MAE={mae:>8.2f}  MAPE={mape_val:>6.2f}%")
    return {"rmse": rmse, "mae": mae, "mape": mape_val}


# ===========================================================================
# 4. Feature Engineering for ML Model
# ===========================================================================
def create_lag_features(df, target_col="sales", lags=None, rolling_windows=None):
    """Create lag and rolling features for the Random Forest model.

    Parameters
    ----------
    df : pd.DataFrame (indexed by date)
    target_col : str
    lags : list[int]
    rolling_windows : list[int]

    Returns
    -------
    df_feat : pd.DataFrame with new columns
    """
    if lags is None:
        lags = [1, 2, 3, 7, 14, 21, 28]
    if rolling_windows is None:
        rolling_windows = [7, 14, 30]

    df_feat = df.copy()

    # Lag features
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)

    # Rolling statistics
    for w in rolling_windows:
        df_feat[f"rolling_mean_{w}"] = df_feat[target_col].shift(1).rolling(window=w).mean()
        df_feat[f"rolling_std_{w}"] = df_feat[target_col].shift(1).rolling(window=w).std()
        df_feat[f"rolling_min_{w}"] = df_feat[target_col].shift(1).rolling(window=w).min()
        df_feat[f"rolling_max_{w}"] = df_feat[target_col].shift(1).rolling(window=w).max()

    # Calendar features
    df_feat["day_of_week"] = df_feat.index.dayofweek
    df_feat["day_of_month"] = df_feat.index.day
    df_feat["month"] = df_feat.index.month
    df_feat["week_of_year"] = df_feat.index.isocalendar().week.astype(int)
    df_feat["is_weekend"] = (df_feat.index.dayofweek >= 5).astype(int)

    # Fourier terms for annual seasonality
    day_of_year = df_feat.index.dayofyear
    for k in [1, 2, 3]:
        df_feat[f"sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        df_feat[f"cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / 365.25)

    df_feat = df_feat.dropna()
    return df_feat


# ===========================================================================
# 5. Models
# ===========================================================================

# --- 5a. ARIMA ---
def fit_arima(train, test, order=(2, 1, 2)):
    """Fit ARIMA and forecast.

    Returns
    -------
    forecast : np.ndarray
    conf_int : pd.DataFrame  (lower, upper)
    """
    if not HAS_STATSMODELS:
        logger.warning("statsmodels unavailable; skipping ARIMA.")
        return None, None

    logger.info(f"Fitting ARIMA{order}...")
    model = ARIMA(train["sales"], order=order)
    fitted = model.fit()

    forecast_result = fitted.get_forecast(steps=len(test))
    forecast = forecast_result.predicted_mean.values
    conf_int = forecast_result.conf_int()
    conf_int.index = test.index

    return forecast, conf_int


# --- 5b. Exponential Smoothing (Holt-Winters) ---
def fit_exponential_smoothing(train, test, seasonal_periods=7):
    """Fit Holt-Winters Exponential Smoothing.

    Returns
    -------
    forecast : np.ndarray
    """
    if not HAS_STATSMODELS:
        logger.warning("statsmodels unavailable; skipping Exponential Smoothing.")
        return None

    logger.info("Fitting Holt-Winters Exponential Smoothing...")
    model = ExponentialSmoothing(
        train["sales"],
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods,
    )
    fitted = model.fit(optimized=True)
    forecast = fitted.forecast(steps=len(test)).values
    return forecast


# --- 5c. Prophet-style Manual Implementation ---
def fit_prophet_manual(train, test):
    """A manual Prophet-like model using trend + Fourier seasonality + regression.

    Components:
        - Piecewise linear trend
        - Annual Fourier seasonality (3 harmonics)
        - Weekly Fourier seasonality (2 harmonics)

    Uses OLS to fit the components.

    Returns
    -------
    forecast : np.ndarray
    """
    logger.info("Fitting Prophet-style manual model (trend + Fourier)...")

    def make_features(dates, reference_date):
        """Build feature matrix of trend + Fourier terms."""
        t = (dates - reference_date).days.values.astype(float)
        doy = dates.dayofyear
        dow = dates.dayofweek

        features = {"t": t, "t_sq": t ** 2}

        # Annual Fourier
        for k in [1, 2, 3]:
            features[f"annual_sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
            features[f"annual_cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)

        # Weekly Fourier
        for k in [1, 2]:
            features[f"weekly_sin_{k}"] = np.sin(2 * np.pi * k * dow / 7)
            features[f"weekly_cos_{k}"] = np.cos(2 * np.pi * k * dow / 7)

        return pd.DataFrame(features, index=dates)

    ref_date = train.index[0]
    X_train = make_features(train.index, ref_date)
    X_test = make_features(test.index, ref_date)

    # Add intercept
    X_train.insert(0, "intercept", 1.0)
    X_test.insert(0, "intercept", 1.0)

    # OLS fit: beta = (X'X)^(-1) X'y
    y_train = train["sales"].values
    XtX = X_train.values.T @ X_train.values
    Xty = X_train.values.T @ y_train
    beta = np.linalg.solve(XtX + 1e-6 * np.eye(XtX.shape[0]), Xty)  # ridge regularization

    forecast = X_test.values @ beta

    # Residual std for confidence intervals
    train_pred = X_train.values @ beta
    residual_std = np.std(y_train - train_pred)

    return forecast, residual_std


# --- 5d. Random Forest with Lag Features ---
def fit_random_forest(df_feat, forecast_horizon):
    """Train a Random Forest regressor on lag features using walk-forward split.

    Parameters
    ----------
    df_feat : pd.DataFrame (with lag features, no NaN)
    forecast_horizon : int

    Returns
    -------
    y_true : np.ndarray  (test period actuals)
    y_pred : np.ndarray  (test period predictions)
    model : RandomForestRegressor
    feature_names : list[str]
    test_index : pd.DatetimeIndex
    """
    logger.info("Fitting Random Forest with lag features...")

    target_col = "sales"
    feature_cols = [c for c in df_feat.columns if c != target_col]

    # Walk-forward: last `forecast_horizon` days as test
    train_data = df_feat.iloc[:-forecast_horizon]
    test_data = df_feat.iloc[-forecast_horizon:]

    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values
    X_test = test_data[feature_cols].values
    y_true = test_data[target_col].values

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_true, y_pred, model, feature_cols, test_data.index


# ===========================================================================
# 6. Walk-Forward Validation
# ===========================================================================
def walk_forward_validation(df, model_type="rf", initial_train_pct=0.7, step=30):
    """Perform walk-forward (expanding window) validation.

    Parameters
    ----------
    df : pd.DataFrame
    model_type : str  ("rf" for Random Forest)
    initial_train_pct : float
    step : int  (number of days to forecast per step)

    Returns
    -------
    all_actuals : list
    all_preds : list
    """
    logger.info(f"Running walk-forward validation (step={step} days)...")
    df_feat = create_lag_features(df)

    target_col = "sales"
    feature_cols = [c for c in df_feat.columns if c != target_col]
    n = len(df_feat)
    initial_train_size = int(n * initial_train_pct)

    all_actuals = []
    all_preds = []
    all_dates = []
    fold = 0

    idx = initial_train_size
    while idx + step <= n:
        fold += 1
        train = df_feat.iloc[:idx]
        test = df_feat.iloc[idx:idx + step]

        model = RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(train[feature_cols].values, train[target_col].values)
        preds = model.predict(test[feature_cols].values)

        all_actuals.extend(test[target_col].values)
        all_preds.extend(preds)
        all_dates.extend(test.index)

        idx += step

    all_actuals = np.array(all_actuals)
    all_preds = np.array(all_preds)

    wf_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    wf_mae = mean_absolute_error(all_actuals, all_preds)
    wf_mape = mape(all_actuals, all_preds)

    print(f"\n  Walk-Forward Validation ({fold} folds):")
    print(f"    RMSE  = {wf_rmse:.2f}")
    print(f"    MAE   = {wf_mae:.2f}")
    print(f"    MAPE  = {wf_mape:.2f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(all_dates, all_actuals, label="Actual", color="steelblue", lw=1)
    ax.plot(all_dates, all_preds, label="Predicted", color="coral", lw=1, alpha=0.8)
    ax.set_title("Walk-Forward Validation (Random Forest)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_walk_forward.png"), dpi=150)
    plt.close()
    logger.info("Saved walk-forward validation plot.")

    return all_actuals, all_preds


# ===========================================================================
# 7. Forecast Visualization
# ===========================================================================
def plot_forecasts(train, test, forecasts, conf_intervals=None):
    """Plot all model forecasts against actuals with confidence intervals.

    Parameters
    ----------
    train : pd.DataFrame
    test : pd.DataFrame
    forecasts : dict  (model_name -> forecast array)
    conf_intervals : dict  (model_name -> (lower, upper) or None)
    """
    if conf_intervals is None:
        conf_intervals = {}

    colors = {
        "ARIMA": "coral",
        "Exp. Smoothing": "seagreen",
        "Prophet-Manual": "darkorange",
        "Random Forest": "purple",
    }

    fig, ax = plt.subplots(figsize=(16, 6))

    # Show last 90 days of training for context
    context_days = 90
    train_context = train.iloc[-context_days:]
    ax.plot(train_context.index, train_context["sales"], color="steelblue", lw=1, label="Training (last 90d)")
    ax.plot(test.index, test["sales"], color="black", lw=2, label="Actual", linestyle="--")

    for name, forecast in forecasts.items():
        if forecast is None:
            continue
        color = colors.get(name, "gray")
        ax.plot(test.index, forecast, color=color, lw=1.5, label=name)

        # Confidence interval
        if name in conf_intervals and conf_intervals[name] is not None:
            lower, upper = conf_intervals[name]
            ax.fill_between(test.index, lower, upper, color=color, alpha=0.15)

    ax.axvline(test.index[0], color="gray", linestyle=":", lw=1, label="Forecast Start")
    ax.set_title("Sales Forecast Comparison", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_forecast_comparison.png"), dpi=150)
    plt.close()
    logger.info("Saved forecast comparison plot.")

    # Individual model plots with confidence intervals
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    for ax, (name, forecast) in zip(axes.flat, forecasts.items()):
        if forecast is None:
            ax.set_title(f"{name} (not available)")
            continue

        color = colors.get(name, "gray")
        ax.plot(test.index, test["sales"], "k--", lw=1.5, label="Actual")
        ax.plot(test.index, forecast, color=color, lw=1.5, label=name)

        if name in conf_intervals and conf_intervals[name] is not None:
            lower, upper = conf_intervals[name]
            ax.fill_between(test.index, lower, upper, color=color, alpha=0.2, label="95% CI")

        rmse_val = np.sqrt(mean_squared_error(test["sales"], forecast))
        ax.set_title(f"{name} (RMSE={rmse_val:.2f})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales ($)")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_individual_forecasts.png"), dpi=150)
    plt.close()
    logger.info("Saved individual forecast plots.")


def plot_feature_importance_rf(model, feature_names, top_n=20):
    """Plot Random Forest feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        [feature_names[i] for i in indices][::-1],
        importances[indices][::-1],
        color="steelblue", edgecolor="black",
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Random Forest - Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_rf_feature_importance.png"), dpi=150)
    plt.close()
    logger.info("Saved RF feature importance plot.")


# ===========================================================================
# 8. Summary
# ===========================================================================
def print_summary(metrics_dict):
    """Print final comparison table."""
    print(f"\n{'=' * 70}")
    print("FINAL MODEL COMPARISON")
    print(f"{'=' * 70}")
    header = f"{'Model':<30} {'RMSE':>10} {'MAE':>10} {'MAPE (%)':>10}"
    print(header)
    print("-" * len(header))

    for name, m in sorted(metrics_dict.items(), key=lambda x: x[1].get("rmse", 999)):
        print(f"{name:<30} {m['rmse']:>10.2f} {m['mae']:>10.2f} {m['mape']:>10.2f}")

    best = min(metrics_dict, key=lambda n: metrics_dict[n].get("rmse", 999))
    print(f"\nBest model by RMSE: {best}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    """Orchestrate the full sales forecasting pipeline."""
    print("\n" + "#" * 70)
    print("#  SALES TIME SERIES FORECASTING")
    print("#" * 70)

    # Step 1 - Generate data
    df = generate_sales_data()

    # Step 2 - EDA & Decomposition
    exploratory_analysis(df)
    decompose_series(df)

    # Step 3 - Train/test split (last FORECAST_HORIZON days as test)
    train = df.iloc[:-FORECAST_HORIZON]
    test = df.iloc[-FORECAST_HORIZON:]
    logger.info(f"Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
    logger.info(f"Test : {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)")

    forecasts = {}
    conf_intervals = {}
    metrics_dict = {}

    print(f"\n{'=' * 70}")
    print("MODEL EVALUATION (Test Period)")
    print(f"{'=' * 70}")

    # --- 5a. ARIMA ---
    arima_forecast, arima_ci = fit_arima(train, test, order=(2, 1, 2))
    if arima_forecast is not None:
        forecasts["ARIMA"] = arima_forecast
        if arima_ci is not None:
            conf_intervals["ARIMA"] = (arima_ci.iloc[:, 0].values, arima_ci.iloc[:, 1].values)
        metrics_dict["ARIMA"] = evaluate_forecast(test["sales"], arima_forecast, "ARIMA")

    # --- 5b. Exponential Smoothing ---
    es_forecast = fit_exponential_smoothing(train, test, seasonal_periods=7)
    if es_forecast is not None:
        forecasts["Exp. Smoothing"] = es_forecast
        # Generate approximate CI from training residual
        es_model = ExponentialSmoothing(
            train["sales"], trend="add", seasonal="add", seasonal_periods=7
        ).fit(optimized=True)
        train_resid_std = np.std(train["sales"].values - es_model.fittedvalues.values)
        conf_intervals["Exp. Smoothing"] = (
            es_forecast - 1.96 * train_resid_std,
            es_forecast + 1.96 * train_resid_std,
        )
        metrics_dict["Exp. Smoothing"] = evaluate_forecast(test["sales"], es_forecast, "Exp. Smoothing")

    # --- 5c. Prophet-style Manual ---
    prophet_forecast, prophet_resid_std = fit_prophet_manual(train, test)
    forecasts["Prophet-Manual"] = prophet_forecast
    conf_intervals["Prophet-Manual"] = (
        prophet_forecast - 1.96 * prophet_resid_std,
        prophet_forecast + 1.96 * prophet_resid_std,
    )
    metrics_dict["Prophet-Manual"] = evaluate_forecast(test["sales"], prophet_forecast, "Prophet-Manual")

    # --- 5d. Random Forest ---
    df_feat = create_lag_features(df)
    rf_actual, rf_pred, rf_model, rf_features, rf_test_idx = fit_random_forest(
        df_feat, FORECAST_HORIZON
    )
    forecasts["Random Forest"] = rf_pred
    # Approximate CI from OOB or training residual
    rf_train_pred = rf_model.predict(
        df_feat.iloc[:-FORECAST_HORIZON][[c for c in df_feat.columns if c != "sales"]].values
    )
    rf_resid_std = np.std(
        df_feat.iloc[:-FORECAST_HORIZON]["sales"].values - rf_train_pred
    )
    conf_intervals["Random Forest"] = (
        rf_pred - 1.96 * rf_resid_std,
        rf_pred + 1.96 * rf_resid_std,
    )
    metrics_dict["Random Forest"] = evaluate_forecast(test["sales"].values[-len(rf_pred):], rf_pred, "Random Forest")

    # Step 6 - Walk-forward validation
    print(f"\n{'=' * 70}")
    print("WALK-FORWARD VALIDATION")
    print(f"{'=' * 70}")
    walk_forward_validation(df, step=30)

    # Step 7 - Visualizations
    plot_forecasts(train, test, forecasts, conf_intervals)
    plot_feature_importance_rf(rf_model, rf_features)

    # Step 8 - Summary
    print_summary(metrics_dict)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
