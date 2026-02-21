"""
Customer Segmentation with Clustering
======================================

This project demonstrates unsupervised learning techniques for customer
segmentation using synthetic retail customer data.

Pipeline stages:
    1. Synthetic data generation (age, income, spending, purchase frequency)
    2. Exploratory data analysis
    3. Feature scaling and PCA for dimensionality reduction
    4. Clustering with K-Means, DBSCAN, and Agglomerative (hierarchical)
    5. Elbow method and silhouette analysis for optimal k
    6. Cluster profiling and business insights
    7. Interactive 3D scatter plots with Plotly

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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Plotly (optional -- falls back to matplotlib if unavailable)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logging.warning(
        "Plotly not installed. 3D plots will use matplotlib instead. "
        "Install with: pip install plotly"
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
N_CUSTOMERS = 2000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
sns.set_style("whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})


# ---------------------------------------------------------------------------
# 1. Synthetic Data Generation
# ---------------------------------------------------------------------------
def generate_customer_data(n=N_CUSTOMERS):
    """Generate synthetic customer data with five natural clusters.

    Clusters roughly represent:
        0 - Young budget shoppers (low income, low spend, high frequency)
        1 - Middle-aged moderate spenders
        2 - High-income, high-spend premium customers
        3 - Older savers (high income, low spend)
        4 - Young high-earners (high income, high spend, high frequency)

    Features
    --------
    - Age : 18-75
    - AnnualIncome : $15k - $200k
    - SpendingScore : 1-100 (retailer-assigned)
    - PurchaseFrequency : purchases per month (1-30)
    - AvgTransactionValue : $ per transaction
    - OnlineSpendRatio : fraction of spend that is online (0-1)

    Returns
    -------
    df : pd.DataFrame
    """
    logger.info(f"Generating synthetic customer data ({n} samples, 5 natural clusters)...")
    rng = np.random.RandomState(RANDOM_STATE)

    cluster_params = [
        # (weight, age_mu, age_sd, income_mu, income_sd, spend_mu, spend_sd,
        #  freq_mu, freq_sd, txn_mu, txn_sd, online_mu, online_sd)
        (0.25, 25, 4, 30000, 8000, 65, 15, 15, 5, 25, 10, 0.70, 0.12),   # young budget
        (0.25, 42, 8, 60000, 15000, 50, 12, 8, 3, 55, 20, 0.45, 0.15),   # mid moderate
        (0.15, 45, 10, 120000, 25000, 85, 8, 10, 4, 120, 30, 0.50, 0.15),  # premium
        (0.20, 58, 7, 90000, 20000, 25, 10, 4, 2, 80, 25, 0.25, 0.10),   # older savers
        (0.15, 30, 5, 100000, 18000, 80, 10, 20, 5, 60, 15, 0.80, 0.08),  # young high
    ]

    dfs = []
    for i, (w, a_m, a_s, i_m, i_s, s_m, s_s, f_m, f_s, t_m, t_s, o_m, o_s) in enumerate(cluster_params):
        size = int(n * w)
        data = pd.DataFrame({
            "Age": rng.normal(a_m, a_s, size).clip(18, 75).astype(int),
            "AnnualIncome": rng.normal(i_m, i_s, size).clip(15000, 200000).round(0),
            "SpendingScore": rng.normal(s_m, s_s, size).clip(1, 100).astype(int),
            "PurchaseFrequency": rng.normal(f_m, f_s, size).clip(1, 30).round(1),
            "AvgTransactionValue": rng.normal(t_m, t_s, size).clip(5, 300).round(2),
            "OnlineSpendRatio": rng.normal(o_m, o_s, size).clip(0, 1).round(3),
            "_TrueCluster": i,
        })
        dfs.append(data)

    df = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    logger.info(f"Dataset shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
def exploratory_analysis(df):
    """Generate EDA plots for the customer dataset."""
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    feature_cols = [c for c in df.columns if c != "_TrueCluster"]
    print(df[feature_cols].describe().round(2).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    feature_cols_plot = [c for c in feature_cols]

    for ax, col in zip(axes.flat, feature_cols_plot):
        ax.hist(df[col], bins=40, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_title(f"Distribution of {col}", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_feature_distributions.png"), dpi=150)
    plt.close()

    # Pairwise scatter (subset of features)
    pair_cols = ["Age", "AnnualIncome", "SpendingScore", "PurchaseFrequency"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    pairs = [(pair_cols[i], pair_cols[j]) for i in range(len(pair_cols)) for j in range(i + 1, len(pair_cols))]

    for ax, (c1, c2) in zip(axes.flat, pairs):
        ax.scatter(df[c1], df[c2], alpha=0.3, s=10, color="steelblue")
        ax.set_xlabel(c1)
        ax.set_ylabel(c2)
        ax.set_title(f"{c1} vs {c2}")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_pairwise_scatter.png"), dpi=150)
    plt.close()

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Feature Correlation Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_correlation_matrix.png"), dpi=150)
    plt.close()

    logger.info("Saved EDA plots.")


# ---------------------------------------------------------------------------
# 3. Preprocessing & PCA
# ---------------------------------------------------------------------------
def preprocess_and_pca(df):
    """Scale features and apply PCA.

    Returns
    -------
    X_scaled : np.ndarray  (scaled original features)
    X_pca : np.ndarray     (first 3 principal components)
    feature_names : list[str]
    scaler : StandardScaler
    pca : PCA
    """
    feature_cols = [c for c in df.columns if c != "_TrueCluster"]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA with 3 components for visualization
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    logger.info(
        f"PCA explained variance: PC1={explained[0]:.2%}, PC2={explained[1]:.2%}, "
        f"PC3={explained[2]:.2%}, total={sum(explained):.2%}"
    )

    # Scree plot
    pca_full = PCA(random_state=RANDOM_STATE).fit(X_scaled)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
                pca_full.explained_variance_ratio_, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Scree Plot")

    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    axes[1].plot(range(1, len(cum_var) + 1), cum_var, "o-", color="steelblue")
    axes[1].axhline(0.90, color="red", linestyle="--", label="90% threshold")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_pca_scree.png"), dpi=150)
    plt.close()
    logger.info("Saved PCA scree plot.")

    return X_scaled, X_pca, feature_cols, scaler, pca


# ---------------------------------------------------------------------------
# 4. Optimal k: Elbow Method & Silhouette Analysis
# ---------------------------------------------------------------------------
def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Run elbow and silhouette analysis to find the optimal number of clusters.

    Parameters
    ----------
    X_scaled : np.ndarray
    k_range : range

    Returns
    -------
    optimal_k : int
    """
    logger.info("Running elbow method and silhouette analysis...")
    inertias = []
    silhouette_scores_list = []
    calinski_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouette_scores_list.append(silhouette_score(X_scaled, labels))
        calinski_scores.append(calinski_harabasz_score(X_scaled, labels))

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Elbow
    axes[0].plot(list(k_range), inertias, "o-", color="steelblue", lw=2)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (Within-cluster SS)")
    axes[0].set_title("Elbow Method")

    # Silhouette
    axes[1].plot(list(k_range), silhouette_scores_list, "o-", color="coral", lw=2)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis")

    # Calinski-Harabasz
    axes[2].plot(list(k_range), calinski_scores, "o-", color="seagreen", lw=2)
    axes[2].set_xlabel("Number of Clusters (k)")
    axes[2].set_ylabel("Calinski-Harabasz Index")
    axes[2].set_title("Calinski-Harabasz Index")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_optimal_k.png"), dpi=150)
    plt.close()

    optimal_k = list(k_range)[np.argmax(silhouette_scores_list)]
    logger.info(f"Optimal k by silhouette score: {optimal_k}")

    # --- Silhouette diagram for optimal k ---
    km = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    sample_scores = silhouette_samples(X_scaled, labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for i in range(optimal_k):
        cluster_scores = np.sort(sample_scores[labels == i])
        size = cluster_scores.shape[0]
        y_upper = y_lower + size

        color = plt.cm.nipy_spectral(float(i) / optimal_k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_scores,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(i), fontsize=10, fontweight="bold")
        y_lower = y_upper + 10

    avg_score = silhouette_score(X_scaled, labels)
    ax.axvline(avg_score, color="red", linestyle="--", lw=1.5, label=f"Avg={avg_score:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette Plot (k={optimal_k})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_silhouette_diagram.png"), dpi=150)
    plt.close()
    logger.info("Saved silhouette diagram.")

    return optimal_k


# ---------------------------------------------------------------------------
# 5. Clustering
# ---------------------------------------------------------------------------
def run_kmeans(X_scaled, k):
    """Run K-Means clustering.

    Returns
    -------
    labels : np.ndarray
    model : KMeans
    """
    logger.info(f"Running K-Means with k={k}...")
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    logger.info(f"K-Means silhouette score: {score:.4f}")
    return labels, km


def run_dbscan(X_scaled):
    """Run DBSCAN clustering with parameter search.

    Returns
    -------
    labels : np.ndarray
    model : DBSCAN
    best_eps : float
    """
    logger.info("Running DBSCAN with eps search...")
    best_score = -1
    best_eps = 1.0
    best_labels = None
    best_model = None

    for eps in np.arange(0.5, 3.0, 0.25):
        db = DBSCAN(eps=eps, min_samples=10)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters >= 2:
            # Exclude noise for silhouette
            mask = labels != -1
            if mask.sum() > n_clusters:
                score = silhouette_score(X_scaled[mask], labels[mask])
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_labels = labels
                    best_model = db

    if best_labels is None:
        logger.warning("DBSCAN could not find >= 2 clusters. Using eps=1.5.")
        best_model = DBSCAN(eps=1.5, min_samples=10)
        best_labels = best_model.fit_predict(X_scaled)

    n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    n_noise = (best_labels == -1).sum()
    logger.info(f"DBSCAN (eps={best_eps:.2f}): {n_clusters} clusters, {n_noise} noise points, "
                f"silhouette={best_score:.4f}")
    return best_labels, best_model, best_eps


def run_hierarchical(X_scaled, k):
    """Run Agglomerative (hierarchical) clustering.

    Returns
    -------
    labels : np.ndarray
    model : AgglomerativeClustering
    """
    logger.info(f"Running Agglomerative Clustering with k={k}...")
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = hc.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    logger.info(f"Hierarchical silhouette score: {score:.4f}")
    return labels, hc


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------
def plot_clusters_2d(X_pca, labels_dict):
    """2D scatter plots of clusters in PCA space."""
    n = len(labels_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, labels) in zip(axes, labels_dict.items()):
        unique_labels = sorted(set(labels))
        palette = plt.cm.nipy_spectral(np.linspace(0, 0.9, max(len(unique_labels), 1)))

        for label, color in zip(unique_labels, palette):
            mask = labels == label
            lbl = f"Noise" if label == -1 else f"Cluster {label}"
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=12, alpha=0.5,
                       color="gray" if label == -1 else color, label=lbl)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{name} Clusters (PCA 2D)")
        ax.legend(fontsize=8, loc="best", markerscale=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_clusters_2d.png"), dpi=150)
    plt.close()
    logger.info("Saved 2D cluster plots.")


def plot_clusters_3d(X_pca, labels, method_name):
    """3D scatter plot using Plotly (or matplotlib fallback)."""
    if HAS_PLOTLY:
        plot_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "PC3": X_pca[:, 2],
            "Cluster": labels.astype(str),
        })
        fig = px.scatter_3d(
            plot_df, x="PC1", y="PC2", z="PC3", color="Cluster",
            title=f"{method_name} Clusters (PCA 3D)",
            opacity=0.6,
            width=900, height=700,
        )
        fig.update_traces(marker=dict(size=3))
        filepath = os.path.join(OUTPUT_DIR, f"08_3d_{method_name.lower().replace(' ', '_')}.html")
        fig.write_html(filepath)
        logger.info(f"Saved interactive 3D plot: {filepath}")
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        unique_labels = sorted(set(labels))
        palette = plt.cm.nipy_spectral(np.linspace(0, 0.9, max(len(unique_labels), 1)))

        for label, color in zip(unique_labels, palette):
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                       s=8, alpha=0.5,
                       color="gray" if label == -1 else color,
                       label=f"Cluster {label}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"{method_name} Clusters (PCA 3D)")
        ax.legend(fontsize=8, markerscale=2)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"08_3d_{method_name.lower().replace(' ', '_')}.png"), dpi=150)
        plt.close()
        logger.info("Saved 3D cluster plot (matplotlib fallback).")


def plot_dendrogram(X_scaled, max_display=30):
    """Plot a truncated dendrogram for hierarchical clustering."""
    logger.info("Computing linkage for dendrogram...")
    # Use a random subset for large datasets
    if X_scaled.shape[0] > 1000:
        idx = np.random.choice(X_scaled.shape[0], 1000, replace=False)
        X_sub = X_scaled[idx]
    else:
        X_sub = X_scaled

    Z = linkage(X_sub, method="ward")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, truncate_mode="lastp", p=max_display, ax=ax,
               leaf_rotation=90, leaf_font_size=8, color_threshold=0)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    ax.set_xlabel("Sample index (or cluster size)")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_dendrogram.png"), dpi=150)
    plt.close()
    logger.info("Saved dendrogram.")


# ---------------------------------------------------------------------------
# 7. Cluster Profiling & Business Insights
# ---------------------------------------------------------------------------
def profile_clusters(df, labels, method_name, feature_cols):
    """Compute and display cluster profiles.

    Parameters
    ----------
    df : pd.DataFrame (original unscaled data)
    labels : np.ndarray
    method_name : str
    feature_cols : list[str]
    """
    df_profiled = df[feature_cols].copy()
    df_profiled["Cluster"] = labels

    # Exclude noise
    df_profiled = df_profiled[df_profiled["Cluster"] != -1]

    profile = df_profiled.groupby("Cluster").agg(["mean", "median", "std", "count"])
    print(f"\n{'=' * 70}")
    print(f"CLUSTER PROFILES ({method_name})")
    print(f"{'=' * 70}")

    cluster_summary = df_profiled.groupby("Cluster")[feature_cols].mean().round(2)
    cluster_counts = df_profiled.groupby("Cluster").size()
    cluster_summary["Size"] = cluster_counts
    cluster_summary["Pct"] = (cluster_counts / cluster_counts.sum() * 100).round(1)
    print(cluster_summary.to_string())

    # --- Radar / polar chart for cluster means ---
    n_clusters = cluster_summary.shape[0]
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5), subplot_kw=dict(polar=True))
    if n_clusters == 1:
        axes = [axes]

    # Normalize means to [0, 1] for radar chart
    means = df_profiled.groupby("Cluster")[feature_cols].mean()
    mins = df_profiled[feature_cols].min()
    maxs = df_profiled[feature_cols].max()
    norm_means = (means - mins) / (maxs - mins + 1e-8)

    angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for ax, (cluster_id, row) in zip(axes, norm_means.iterrows()):
        values = row.tolist() + [row.tolist()[0]]
        ax.fill(angles, values, alpha=0.25, color=plt.cm.nipy_spectral(cluster_id / n_clusters))
        ax.plot(angles, values, "o-", lw=2, color=plt.cm.nipy_spectral(cluster_id / n_clusters))
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_cols, fontsize=7)
        ax.set_title(f"Cluster {cluster_id} (n={int(cluster_summary.loc[cluster_id, 'Size'])})",
                     fontsize=10, pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"10_radar_{method_name.lower().replace(' ', '_')}.png"), dpi=150)
    plt.close()
    logger.info("Saved radar profiles.")

    # --- Box plots per feature per cluster ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, col in zip(axes.flat, feature_cols):
        df_profiled.boxplot(column=col, by="Cluster", ax=ax)
        ax.set_title(f"{col} by Cluster")
        ax.set_xlabel("Cluster")
    plt.suptitle(f"Feature Distributions by Cluster ({method_name})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"11_boxplots_{method_name.lower().replace(' ', '_')}.png"), dpi=150)
    plt.close()
    logger.info("Saved box plots.")

    # --- Business insights ---
    print(f"\n--- Business Insights ({method_name}) ---")
    for cid in sorted(df_profiled["Cluster"].unique()):
        c = cluster_summary.loc[cid]
        print(f"\n  Cluster {cid} ({int(c['Size'])} customers, {c['Pct']}%):")

        # Characterize based on relative feature values
        insights = []
        for feat in feature_cols:
            overall_mean = df_profiled[feat].mean()
            cluster_mean = means.loc[cid, feat]
            ratio = cluster_mean / (overall_mean + 1e-8)

            if ratio > 1.2:
                insights.append(f"    - High {feat} ({cluster_mean:.1f} vs avg {overall_mean:.1f})")
            elif ratio < 0.8:
                insights.append(f"    - Low {feat} ({cluster_mean:.1f} vs avg {overall_mean:.1f})")

        if insights:
            print("\n".join(insights))
        else:
            print("    - Close to average across all features")

    return cluster_summary


# ---------------------------------------------------------------------------
# 8. Comparison Summary
# ---------------------------------------------------------------------------
def compare_methods(X_scaled, labels_dict):
    """Print a comparison table of clustering methods."""
    print(f"\n{'=' * 70}")
    print("CLUSTERING METHOD COMPARISON")
    print(f"{'=' * 70}")
    header = f"{'Method':<25} {'Clusters':>9} {'Noise':>7} {'Silhouette':>12} {'Calinski-H':>12}"
    print(header)
    print("-" * len(header))

    for name, labels in labels_dict.items():
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        mask = labels != -1
        if mask.sum() > n_clusters and n_clusters >= 2:
            sil = silhouette_score(X_scaled[mask], labels[mask])
            cal = calinski_harabasz_score(X_scaled[mask], labels[mask])
        else:
            sil = float("nan")
            cal = float("nan")

        print(f"{name:<25} {n_clusters:>9} {n_noise:>7} {sil:>12.4f} {cal:>12.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Orchestrate the customer segmentation pipeline."""
    print("\n" + "#" * 70)
    print("#  CUSTOMER SEGMENTATION - CLUSTERING ANALYSIS")
    print("#" * 70)

    # Step 1 - Data generation
    df = generate_customer_data()

    # Step 2 - EDA
    exploratory_analysis(df)

    # Step 3 - Preprocessing & PCA
    feature_cols = [c for c in df.columns if c != "_TrueCluster"]
    X_scaled, X_pca, feature_cols, scaler, pca = preprocess_and_pca(df)

    # Step 4 - Find optimal k
    optimal_k = find_optimal_k(X_scaled)

    # Step 5 - Clustering
    kmeans_labels, kmeans_model = run_kmeans(X_scaled, optimal_k)
    dbscan_labels, dbscan_model, best_eps = run_dbscan(X_scaled)
    hier_labels, hier_model = run_hierarchical(X_scaled, optimal_k)

    labels_dict = {
        "K-Means": kmeans_labels,
        "DBSCAN": dbscan_labels,
        "Hierarchical": hier_labels,
    }

    # Step 6 - Visualization
    plot_clusters_2d(X_pca, labels_dict)
    plot_clusters_3d(X_pca, kmeans_labels, "K-Means")
    plot_clusters_3d(X_pca, hier_labels, "Hierarchical")
    plot_dendrogram(X_scaled)

    # Step 7 - Cluster profiling
    print("\n")
    profile_clusters(df, kmeans_labels, "K-Means", feature_cols)

    # Step 8 - Method comparison
    compare_methods(X_scaled, labels_dict)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
