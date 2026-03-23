#!/usr/bin/env python3
"""
Netflix Content Exploratory Data Analysis
==========================================
Generates a synthetic Netflix-like dataset and performs comprehensive EDA
including content distribution, temporal trends, and geographic analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_TITLES = 1000
OUTPUT_DIR = "outputs"

np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)


# ===========================================================================
# 1. Synthetic Data Generation
# ===========================================================================
def generate_netflix_data(n: int = N_TITLES) -> pd.DataFrame:
    """Create a synthetic Netflix-style catalogue."""

    types = np.random.choice(["Movie", "TV Show"], size=n, p=[0.68, 0.32])

    genres = [
        "Drama", "Comedy", "Action", "Thriller", "Documentary",
        "Horror", "Romance", "Sci-Fi", "Animation", "Crime",
        "Fantasy", "Family", "Mystery", "Adventure", "Musical",
    ]
    genre_weights = np.array([18, 15, 12, 10, 9, 7, 6, 5, 5, 4, 3, 2, 2, 1, 1],
                             dtype=float)
    genre_weights /= genre_weights.sum()
    primary_genre = np.random.choice(genres, size=n, p=genre_weights)

    years = np.random.choice(
        range(2000, 2025),
        size=n,
        p=_year_weights(2000, 2024),
    )

    durations = []
    for t in types:
        if t == "Movie":
            durations.append(int(np.random.normal(105, 25)))
        else:
            durations.append(int(np.random.choice(range(1, 9), p=_season_weights())))

    countries = [
        "United States", "India", "United Kingdom", "Japan", "South Korea",
        "Canada", "France", "Germany", "Spain", "Brazil",
        "Mexico", "Australia", "Nigeria", "Turkey", "Thailand",
    ]
    country_weights = np.array([30, 14, 9, 7, 6, 5, 4, 4, 3, 3, 3, 3, 3, 3, 3],
                               dtype=float)
    country_weights /= country_weights.sum()
    country_col = np.random.choice(countries, size=n, p=country_weights)

    ratings = ["TV-MA", "TV-14", "TV-PG", "R", "PG-13", "PG", "TV-Y7", "TV-G", "G", "NR"]
    rating_weights = np.array([25, 22, 12, 10, 9, 7, 5, 4, 3, 3], dtype=float)
    rating_weights /= rating_weights.sum()
    rating_col = np.random.choice(ratings, size=n, p=rating_weights)

    # Build a secondary genre for richer analysis
    secondary_genre = np.random.choice(genres, size=n)

    df = pd.DataFrame({
        "title_id": range(1, n + 1),
        "type": types,
        "genre": primary_genre,
        "secondary_genre": secondary_genre,
        "release_year": years,
        "duration": durations,
        "country": country_col,
        "rating": rating_col,
        "imdb_score": np.round(np.clip(np.random.normal(6.5, 1.2, n), 1, 10), 1),
    })

    # Label duration column properly
    df["duration_label"] = df.apply(
        lambda r: f"{r['duration']} min" if r["type"] == "Movie"
        else f"{r['duration']} Season(s)", axis=1,
    )

    return df


def _year_weights(start: int, end: int) -> np.ndarray:
    """Exponential-ish growth toward recent years."""
    years = np.arange(start, end + 1)
    w = np.exp(0.12 * (years - start))
    return w / w.sum()


def _season_weights() -> list:
    """Probability distribution over number of seasons (1-8)."""
    w = np.array([40, 25, 15, 8, 5, 3, 2, 2], dtype=float)
    return (w / w.sum()).tolist()


# ===========================================================================
# 2. Exploratory Data Analysis
# ===========================================================================
def basic_overview(df: pd.DataFrame) -> None:
    """Print dataset summary statistics."""
    print("=" * 70)
    print("NETFLIX DATASET — BASIC OVERVIEW")
    print("=" * 70)
    print(f"\nShape : {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    print(df.dtypes)
    print("\n--- Descriptive Statistics (numeric) ---")
    print(df.describe())
    print("\n--- Value Counts: type ---")
    print(df["type"].value_counts())
    print("\n--- Missing values ---")
    print(df.isnull().sum())
    print()


def content_distribution_analysis(df: pd.DataFrame) -> None:
    """Analyse content distribution by type, genre, and rating."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Content Distribution Analysis", fontsize=18, fontweight="bold", y=1.02)

    # 1. Type split
    type_counts = df["type"].value_counts()
    axes[0, 0].pie(
        type_counts,
        labels=type_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("Set2", len(type_counts)),
        explode=[0.04] * len(type_counts),
        textprops={"fontsize": 12},
    )
    axes[0, 0].set_title("Movies vs TV Shows", fontsize=14)

    # 2. Top genres
    genre_order = df["genre"].value_counts().head(10).index
    sns.countplot(data=df, y="genre", order=genre_order, hue="type", ax=axes[0, 1])
    axes[0, 1].set_title("Top 10 Genres by Content Type", fontsize=14)
    axes[0, 1].set_xlabel("Count")
    axes[0, 1].set_ylabel("")

    # 3. Rating distribution
    rating_order = df["rating"].value_counts().index
    sns.countplot(data=df, x="rating", order=rating_order, hue="type", ax=axes[1, 0])
    axes[1, 0].set_title("Rating Distribution", fontsize=14)
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 4. IMDB score distribution
    sns.histplot(data=df, x="imdb_score", hue="type", kde=True, bins=20, ax=axes[1, 1])
    axes[1, 1].set_title("IMDB Score Distribution", fontsize=14)
    axes[1, 1].set_xlabel("IMDB Score")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/content_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] content_distribution.png")


def temporal_trends_analysis(df: pd.DataFrame) -> None:
    """Analyse how Netflix content has changed over time."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Temporal Trends Analysis", fontsize=18, fontweight="bold", y=1.02)

    # 1. Titles added per year
    year_type = df.groupby(["release_year", "type"]).size().reset_index(name="count")
    for t, grp in year_type.groupby("type"):
        axes[0, 0].plot(grp["release_year"], grp["count"], marker="o", label=t, linewidth=2)
    axes[0, 0].set_title("Titles Released per Year", fontsize=14)
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("Number of Titles")
    axes[0, 0].legend()

    # 2. Average IMDB score over time
    yearly_score = df.groupby("release_year")["imdb_score"].mean()
    axes[0, 1].plot(yearly_score.index, yearly_score.values, marker="s",
                    color="coral", linewidth=2)
    axes[0, 1].fill_between(yearly_score.index, yearly_score.values, alpha=0.2, color="coral")
    axes[0, 1].set_title("Average IMDB Score Over Time", fontsize=14)
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].set_ylabel("Mean IMDB Score")

    # 3. Genre popularity over 5-year bins
    df_copy = df.copy()
    df_copy["year_bin"] = pd.cut(df_copy["release_year"],
                                  bins=range(2000, 2030, 5), right=False)
    top5_genres = df["genre"].value_counts().head(5).index
    genre_time = (
        df_copy[df_copy["genre"].isin(top5_genres)]
        .groupby(["year_bin", "genre"])
        .size()
        .reset_index(name="count")
    )
    genre_time["year_bin"] = genre_time["year_bin"].astype(str)
    sns.barplot(data=genre_time, x="year_bin", y="count", hue="genre", ax=axes[1, 0])
    axes[1, 0].set_title("Genre Popularity Over Time (5-year bins)", fontsize=14)
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].set_xlabel("")

    # 4. Movie duration trend
    movies = df[df["type"] == "Movie"].copy()
    dur_by_year = movies.groupby("release_year")["duration"].agg(["mean", "std"]).dropna()
    axes[1, 1].plot(dur_by_year.index, dur_by_year["mean"], marker="D",
                    color="steelblue", linewidth=2)
    axes[1, 1].fill_between(
        dur_by_year.index,
        dur_by_year["mean"] - dur_by_year["std"],
        dur_by_year["mean"] + dur_by_year["std"],
        alpha=0.2, color="steelblue",
    )
    axes[1, 1].set_title("Average Movie Duration Over Time", fontsize=14)
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Duration (min)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/temporal_trends.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] temporal_trends.png")


def geographic_analysis(df: pd.DataFrame) -> None:
    """Analyse content production across countries."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Geographic Analysis", fontsize=18, fontweight="bold", y=1.02)

    # 1. Top producing countries
    country_counts = df["country"].value_counts().head(10)
    sns.barplot(x=country_counts.values, y=country_counts.index,
                palette="viridis", ax=axes[0, 0])
    axes[0, 0].set_title("Top 10 Content-Producing Countries", fontsize=14)
    axes[0, 0].set_xlabel("Number of Titles")

    # 2. Content type by country
    top5_countries = df["country"].value_counts().head(5).index
    ct = df[df["country"].isin(top5_countries)].groupby(["country", "type"]).size().unstack(fill_value=0)
    ct.plot(kind="barh", stacked=True, ax=axes[0, 1], colormap="Set2")
    axes[0, 1].set_title("Content Type Split — Top 5 Countries", fontsize=14)
    axes[0, 1].set_xlabel("Count")
    axes[0, 1].set_ylabel("")

    # 3. Average IMDB score by country
    country_score = (
        df.groupby("country")["imdb_score"]
        .agg(["mean", "count"])
        .query("count >= 10")
        .sort_values("mean", ascending=False)
        .head(10)
    )
    sns.barplot(x=country_score["mean"].values, y=country_score.index,
                palette="magma", ax=axes[1, 0])
    axes[1, 0].set_title("Avg IMDB Score by Country (min 10 titles)", fontsize=14)
    axes[1, 0].set_xlabel("Mean IMDB Score")
    axes[1, 0].set_xlim(5, 8)

    # 4. Genre preference heatmap by top countries
    top_countries = df["country"].value_counts().head(8).index
    top_genres = df["genre"].value_counts().head(8).index
    heatmap_data = (
        df[df["country"].isin(top_countries) & df["genre"].isin(top_genres)]
        .groupby(["country", "genre"])
        .size()
        .unstack(fill_value=0)
    )
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd", ax=axes[1, 1])
    axes[1, 1].set_title("Genre × Country Heatmap", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/geographic_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] geographic_analysis.png")


def advanced_analysis(df: pd.DataFrame) -> None:
    """Correlation matrix and cross-tabulation deep dive."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Advanced Analysis", fontsize=18, fontweight="bold", y=1.02)

    # 1. Correlation of numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0,
                fmt=".2f", ax=axes[0], square=True)
    axes[0].set_title("Correlation Matrix", fontsize=14)

    # 2. Box plot: IMDB score by genre (top 8)
    top8_genres = df["genre"].value_counts().head(8).index
    sns.boxplot(data=df[df["genre"].isin(top8_genres)],
                x="genre", y="imdb_score", palette="Set3", ax=axes[1])
    axes[1].set_title("IMDB Score by Genre", fontsize=14)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/advanced_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] advanced_analysis.png")


def print_key_insights(df: pd.DataFrame) -> None:
    """Summarise key insights from the data."""
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    movies = df[df["type"] == "Movie"]
    shows = df[df["type"] == "TV Show"]
    print(f"  Total titles        : {len(df)}")
    print(f"  Movies / TV Shows   : {len(movies)} / {len(shows)}")
    print(f"  Unique countries    : {df['country'].nunique()}")
    print(f"  Year range          : {df['release_year'].min()} – {df['release_year'].max()}")
    print(f"  Mean IMDB (Movie)   : {movies['imdb_score'].mean():.2f}")
    print(f"  Mean IMDB (TV Show) : {shows['imdb_score'].mean():.2f}")
    print(f"  Top genre           : {df['genre'].value_counts().idxmax()}")
    print(f"  Top country         : {df['country'].value_counts().idxmax()}")
    top_rated = df.nlargest(3, "imdb_score")[["title_id", "genre", "imdb_score", "release_year"]]
    print(f"\n  Top 3 highest-rated titles:\n{top_rated.to_string(index=False)}")
    print("=" * 70)


# ===========================================================================
# 3. Main
# ===========================================================================
def main() -> None:
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating synthetic Netflix dataset ...")
    df = generate_netflix_data()
    df.to_csv(f"{OUTPUT_DIR}/netflix_synthetic.csv", index=False)
    print(f"[Saved] netflix_synthetic.csv  ({len(df)} rows)\n")

    basic_overview(df)
    content_distribution_analysis(df)
    temporal_trends_analysis(df)
    geographic_analysis(df)
    advanced_analysis(df)
    print_key_insights(df)

    print("\nAll analyses complete. Charts saved to the 'outputs/' directory.")


if __name__ == "__main__":
    main()
