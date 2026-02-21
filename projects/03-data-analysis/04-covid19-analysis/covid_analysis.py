#!/usr/bin/env python3
"""
Epidemic / COVID-19 Style Data Analysis
========================================
Generates synthetic epidemic data for multiple countries over 365 days
and performs time series analysis including CFR, growth rates, and
moving averages.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_DAYS = 365
OUTPUT_DIR = "outputs"

np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)

COUNTRIES = {
    "United States": {"pop": 331_000_000, "peak_day": 90, "severity": 1.0},
    "Brazil":        {"pop": 213_000_000, "peak_day": 120, "severity": 0.85},
    "India":         {"pop": 1_380_000_000, "peak_day": 150, "severity": 0.70},
    "Germany":       {"pop": 83_000_000, "peak_day": 80, "severity": 0.55},
    "Japan":         {"pop": 126_000_000, "peak_day": 110, "severity": 0.40},
    "South Africa":  {"pop": 60_000_000, "peak_day": 140, "severity": 0.65},
    "United Kingdom": {"pop": 67_000_000, "peak_day": 85, "severity": 0.75},
    "Mexico":        {"pop": 128_000_000, "peak_day": 130, "severity": 0.80},
}


# ===========================================================================
# 1. Synthetic Data Generation
# ===========================================================================
def generate_epidemic_data(n_days: int = N_DAYS) -> pd.DataFrame:
    """
    Simulate an epidemic curve for each country using a modified
    Gaussian wave model with noise.
    """
    rows = []
    start_date = datetime(2023, 1, 1)

    for country, params in COUNTRIES.items():
        pop = params["pop"]
        peak = params["peak_day"]
        severity = params["severity"]

        # Generate daily new cases with a wave shape
        days = np.arange(n_days)

        # Primary wave (Gaussian)
        wave1 = severity * 50000 * np.exp(-0.5 * ((days - peak) / 30) ** 2)
        # Secondary smaller wave
        wave2 = severity * 20000 * np.exp(-0.5 * ((days - peak - 150) / 25) ** 2)
        daily_cases = wave1 + wave2

        # Add noise
        noise = np.random.normal(0, severity * 2000, n_days)
        daily_cases = np.maximum(daily_cases + noise, 0).astype(int)

        # CFR varies by country (1-4%)
        base_cfr = np.random.uniform(0.01, 0.04)

        # Daily deaths lag cases by ~14 days
        daily_deaths = np.zeros(n_days, dtype=int)
        for d in range(n_days):
            lag_day = max(0, d - 14)
            daily_deaths[d] = max(0, int(daily_cases[lag_day] * base_cfr *
                                         np.random.normal(1, 0.15)))

        # Recovered: ~97% of cases, lagged by 21 days
        daily_recovered = np.zeros(n_days, dtype=int)
        for d in range(n_days):
            lag_day = max(0, d - 21)
            daily_recovered[d] = max(0, int(daily_cases[lag_day] * 0.97 *
                                            np.random.normal(1, 0.1)))

        # Tests: proportional to cases with a floor
        daily_tests = (daily_cases * np.random.uniform(8, 15, n_days) +
                       np.random.uniform(5000, 30000, n_days)).astype(int)

        cumulative_cases = np.cumsum(daily_cases)
        cumulative_deaths = np.cumsum(daily_deaths)
        cumulative_recovered = np.cumsum(daily_recovered)

        for d in range(n_days):
            rows.append({
                "date": start_date + timedelta(days=int(d)),
                "country": country,
                "population": pop,
                "daily_cases": daily_cases[d],
                "daily_deaths": daily_deaths[d],
                "daily_recovered": daily_recovered[d],
                "daily_tests": daily_tests[d],
                "cumulative_cases": cumulative_cases[d],
                "cumulative_deaths": cumulative_deaths[d],
                "cumulative_recovered": cumulative_recovered[d],
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # Derived metrics
    df["positivity_rate"] = (df["daily_cases"] / df["daily_tests"].replace(0, 1)).clip(0, 1).round(4)
    df["active_cases"] = (df["cumulative_cases"] - df["cumulative_deaths"] -
                          df["cumulative_recovered"]).clip(lower=0)
    df["cases_per_million"] = (df["daily_cases"] / (df["population"] / 1e6)).round(2)
    df["deaths_per_million"] = (df["daily_deaths"] / (df["population"] / 1e6)).round(4)

    return df


# ===========================================================================
# 2. Exploratory Data Analysis
# ===========================================================================
def basic_overview(df: pd.DataFrame) -> None:
    """Print dataset summary."""
    print("=" * 70)
    print("EPIDEMIC DATASET — BASIC OVERVIEW")
    print("=" * 70)
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"\n--- Total Cases & Deaths by Country ---")
    summary = (
        df.groupby("country")
        .agg(
            total_cases=("cumulative_cases", "max"),
            total_deaths=("cumulative_deaths", "max"),
            peak_daily_cases=("daily_cases", "max"),
        )
        .sort_values("total_cases", ascending=False)
    )
    summary["cfr_pct"] = (summary["total_deaths"] / summary["total_cases"] * 100).round(2)
    print(summary)
    print()


def time_series_analysis(df: pd.DataFrame) -> None:
    """Visualise daily cases, deaths, and trends over time."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Time Series Analysis", fontsize=18, fontweight="bold", y=1.02)

    # 1. Daily new cases by country
    for country in COUNTRIES:
        cdf = df[df["country"] == country]
        axes[0, 0].plot(cdf["date"], cdf["daily_cases"], label=country, linewidth=1.2)
    axes[0, 0].set_title("Daily New Cases", fontsize=14)
    axes[0, 0].set_ylabel("Cases")
    axes[0, 0].legend(fontsize=8, ncol=2)

    # 2. Daily deaths by country
    for country in COUNTRIES:
        cdf = df[df["country"] == country]
        axes[0, 1].plot(cdf["date"], cdf["daily_deaths"], label=country, linewidth=1.2)
    axes[0, 1].set_title("Daily Deaths", fontsize=14)
    axes[0, 1].set_ylabel("Deaths")
    axes[0, 1].legend(fontsize=8, ncol=2)

    # 3. Cumulative cases
    for country in COUNTRIES:
        cdf = df[df["country"] == country]
        axes[1, 0].plot(cdf["date"], cdf["cumulative_cases"], label=country, linewidth=1.5)
    axes[1, 0].set_title("Cumulative Cases", fontsize=14)
    axes[1, 0].set_ylabel("Total Cases")
    axes[1, 0].legend(fontsize=8, ncol=2)

    # 4. Cases per million (normalised)
    for country in COUNTRIES:
        cdf = df[df["country"] == country]
        axes[1, 1].plot(cdf["date"], cdf["cases_per_million"], label=country, linewidth=1.2)
    axes[1, 1].set_title("Daily Cases per Million", fontsize=14)
    axes[1, 1].set_ylabel("Cases per Million")
    axes[1, 1].legend(fontsize=8, ncol=2)

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/time_series.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] time_series.png")


def cfr_analysis(df: pd.DataFrame) -> None:
    """Case Fatality Rate analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Case Fatality Rate (CFR) Analysis", fontsize=18, fontweight="bold", y=1.02)

    # 1. Overall CFR by country
    country_stats = df.groupby("country").agg(
        total_cases=("cumulative_cases", "max"),
        total_deaths=("cumulative_deaths", "max"),
    )
    country_stats["cfr"] = (country_stats["total_deaths"] /
                            country_stats["total_cases"] * 100).round(2)
    country_stats = country_stats.sort_values("cfr", ascending=True)

    sns.barplot(x=country_stats["cfr"].values, y=country_stats.index,
                palette="Reds", ax=axes[0])
    axes[0].set_title("Overall CFR by Country (%)", fontsize=14)
    axes[0].set_xlabel("CFR (%)")
    for i, v in enumerate(country_stats["cfr"].values):
        axes[0].text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=10)

    # 2. Rolling CFR over time for top 4 countries
    top4 = (df.groupby("country")["cumulative_cases"].max()
            .nlargest(4).index.tolist())
    for country in top4:
        cdf = df[df["country"] == country].copy()
        cdf["rolling_cfr"] = (
            cdf["cumulative_deaths"] / cdf["cumulative_cases"].replace(0, 1) * 100
        )
        axes[1].plot(cdf["date"], cdf["rolling_cfr"], label=country, linewidth=1.5)
    axes[1].set_title("Rolling CFR Over Time", fontsize=14)
    axes[1].set_ylabel("CFR (%)")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cfr_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] cfr_analysis.png")


def growth_rate_analysis(df: pd.DataFrame) -> None:
    """Calculate and visualise growth rates and doubling times."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Growth Rate & Moving Average Analysis", fontsize=18, fontweight="bold", y=1.02)

    top4 = (df.groupby("country")["cumulative_cases"].max()
            .nlargest(4).index.tolist())

    # 1. 7-day moving average of daily cases
    for country in top4:
        cdf = df[df["country"] == country].copy()
        cdf["ma7"] = cdf["daily_cases"].rolling(7).mean()
        axes[0, 0].plot(cdf["date"], cdf["ma7"], label=country, linewidth=1.5)
    axes[0, 0].set_title("7-Day Moving Average (Cases)", fontsize=14)
    axes[0, 0].set_ylabel("Cases (7-day MA)")
    axes[0, 0].legend()

    # 2. 14-day moving average of daily deaths
    for country in top4:
        cdf = df[df["country"] == country].copy()
        cdf["ma14_deaths"] = cdf["daily_deaths"].rolling(14).mean()
        axes[0, 1].plot(cdf["date"], cdf["ma14_deaths"], label=country, linewidth=1.5)
    axes[0, 1].set_title("14-Day Moving Average (Deaths)", fontsize=14)
    axes[0, 1].set_ylabel("Deaths (14-day MA)")
    axes[0, 1].legend()

    # 3. Weekly growth rate
    for country in top4:
        cdf = df[df["country"] == country].copy()
        weekly = cdf.set_index("date")["daily_cases"].resample("W").sum()
        growth = weekly.pct_change().clip(-1, 5) * 100
        axes[1, 0].plot(growth.index, growth.values, label=country,
                        linewidth=1.2, alpha=0.8)
    axes[1, 0].axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 0].set_title("Weekly Case Growth Rate (%)", fontsize=14)
    axes[1, 0].set_ylabel("Growth Rate (%)")
    axes[1, 0].legend()

    # 4. Positivity rate
    for country in top4:
        cdf = df[df["country"] == country].copy()
        cdf["pos_ma7"] = cdf["positivity_rate"].rolling(7).mean() * 100
        axes[1, 1].plot(cdf["date"], cdf["pos_ma7"], label=country, linewidth=1.5)
    axes[1, 1].axhline(y=5, color="red", linewidth=1, linestyle="--",
                       label="5% Threshold")
    axes[1, 1].set_title("Test Positivity Rate (7-day MA)", fontsize=14)
    axes[1, 1].set_ylabel("Positivity (%)")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/growth_rates.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] growth_rates.png")


def comparative_analysis(df: pd.DataFrame) -> None:
    """Cross-country comparison charts."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Comparative Country Analysis", fontsize=18, fontweight="bold", y=1.02)

    country_totals = df.groupby("country").agg(
        total_cases=("cumulative_cases", "max"),
        total_deaths=("cumulative_deaths", "max"),
        total_tests=("daily_tests", "sum"),
        peak_cases=("daily_cases", "max"),
        population=("population", "first"),
    ).sort_values("total_cases", ascending=False)

    # 1. Total cases bar chart
    sns.barplot(x=country_totals["total_cases"].values,
                y=country_totals.index, palette="Blues_d", ax=axes[0, 0])
    axes[0, 0].set_title("Total Cumulative Cases", fontsize=14)
    axes[0, 0].set_xlabel("Cases")

    # 2. Total deaths bar chart
    sns.barplot(x=country_totals["total_deaths"].values,
                y=country_totals.index, palette="Reds_d", ax=axes[0, 1])
    axes[0, 1].set_title("Total Deaths", fontsize=14)
    axes[0, 1].set_xlabel("Deaths")

    # 3. Cases per million total
    country_totals["cases_per_mil"] = (
        country_totals["total_cases"] / (country_totals["population"] / 1e6)
    ).round(0)
    cpm = country_totals.sort_values("cases_per_mil", ascending=True)
    sns.barplot(x=cpm["cases_per_mil"].values, y=cpm.index,
                palette="Oranges", ax=axes[1, 0])
    axes[1, 0].set_title("Total Cases per Million", fontsize=14)
    axes[1, 0].set_xlabel("Cases per Million")

    # 4. Peak daily cases
    peak_sorted = country_totals.sort_values("peak_cases", ascending=True)
    sns.barplot(x=peak_sorted["peak_cases"].values, y=peak_sorted.index,
                palette="Purples", ax=axes[1, 1])
    axes[1, 1].set_title("Peak Daily Cases", fontsize=14)
    axes[1, 1].set_xlabel("Peak Cases")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparative_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] comparative_analysis.png")


def active_cases_analysis(df: pd.DataFrame) -> None:
    """Analyse active cases over time."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for country in COUNTRIES:
        cdf = df[df["country"] == country]
        ax.plot(cdf["date"], cdf["active_cases"], label=country, linewidth=1.3)

    ax.set_title("Active Cases Over Time", fontsize=16, fontweight="bold")
    ax.set_ylabel("Active Cases")
    ax.legend(fontsize=9, ncol=2)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/active_cases.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] active_cases.png")


def print_key_insights(df: pd.DataFrame) -> None:
    """Print key findings."""
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    totals = df.groupby("country").agg(
        total_cases=("cumulative_cases", "max"),
        total_deaths=("cumulative_deaths", "max"),
        peak_daily=("daily_cases", "max"),
        population=("population", "first"),
    )
    totals["cfr"] = (totals["total_deaths"] / totals["total_cases"] * 100).round(2)
    totals["cases_per_mil"] = (totals["total_cases"] / (totals["population"] / 1e6)).round(0)

    most_cases = totals["total_cases"].idxmax()
    highest_cfr = totals["cfr"].idxmax()
    highest_cpm = totals["cases_per_mil"].idxmax()

    print(f"  Most total cases         : {most_cases} ({totals.loc[most_cases, 'total_cases']:,})")
    print(f"  Highest CFR              : {highest_cfr} ({totals.loc[highest_cfr, 'cfr']:.2f}%)")
    print(f"  Highest cases/million    : {highest_cpm} ({totals.loc[highest_cpm, 'cases_per_mil']:,.0f})")
    print(f"  Global total cases       : {totals['total_cases'].sum():,}")
    print(f"  Global total deaths      : {totals['total_deaths'].sum():,}")
    print(f"  Global average CFR       : {totals['total_deaths'].sum() / totals['total_cases'].sum() * 100:.2f}%")
    print("=" * 70)


# ===========================================================================
# 3. Main
# ===========================================================================
def main() -> None:
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating synthetic epidemic dataset ...")
    df = generate_epidemic_data()
    df.to_csv(f"{OUTPUT_DIR}/epidemic_synthetic.csv", index=False)
    print(f"[Saved] epidemic_synthetic.csv  ({len(df)} rows)\n")

    basic_overview(df)
    time_series_analysis(df)
    cfr_analysis(df)
    growth_rate_analysis(df)
    comparative_analysis(df)
    active_cases_analysis(df)
    print_key_insights(df)

    print("\nAll analyses complete. Charts saved to the 'outputs/' directory.")


if __name__ == "__main__":
    main()
