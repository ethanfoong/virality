import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def find_features_complete(project_root: Path) -> Path:
    candidates = [
        project_root / "features_complete.csv",
        project_root / "data" / "processed" / "features_complete.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find features_complete.csv")


def select_cluster_features(df: pd.DataFrame):
    # Text PCA components
    pca_cols = [c for c in df.columns if c.lower().startswith("headline_pc")]

    # Other engineered features (only if they exist)
    candidate_other = [
        "keyword_density",
        "content_richness",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "positive_strength",
        "negative_strength",
        "sentiment_balance",
        "pos_neg_ratio",
    ]
    other_cols = [c for c in candidate_other if c in df.columns]

    feature_cols = pca_cols + other_cols
    if not feature_cols:
        raise ValueError("No clustering features found; check column names")

    print("\nUsing the following columns for clustering:")
    for c in feature_cols:
        print("  -", c)

    X = df[feature_cols].fillna(0)

    return X, feature_cols


def compute_kmeans_metrics(X_scaled, k_range):
    silhouette_scores = {}
    inertia_values = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        inertia_values[k] = km.inertia_
        silhouette_scores[k] = silhouette_score(X_scaled, labels)

        print(f"k={k}: silhouette={silhouette_scores[k]:.4f}, inertia={inertia_values[k]:.0f}")

    return silhouette_scores, inertia_values


def plot_silhouette(scores, fig_path):
    plt.figure(figsize=(6, 4))
    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]

    plt.plot(ks, vals, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette scores for KMeans clustering")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print("Saved silhouette plot:", fig_path)


def plot_elbow(inertia, fig_path):
    plt.figure(figsize=(6, 4))
    ks = sorted(inertia.keys())
    vals = [inertia[k] for k in ks]

    plt.plot(ks, vals, marker="o", color="darkred")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (Within-cluster SSE)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print("Saved elbow plot:", fig_path)


def plot_pca_clusters(X_scaled, labels, fig_path):
    pca2 = PCA(n_components=2, random_state=42)
    coords = pca2.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels,
                    palette="tab10", alpha=0.5, s=15, legend="full")
    plt.title("KMeans Clusters (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print("Saved PCA cluster plot:", fig_path)


def plot_cluster_bars(df, fig_dir):
    if "viral" in df.columns:
        plt.figure(figsize=(6, 4))
        df.groupby("cluster_kmeans")["viral"].mean().plot(kind="bar")
        plt.title("Viral Rate by Cluster")
        plt.tight_layout()
        p = fig_dir / "cluster_viral_rate.png"
        plt.savefig(p)
        plt.close()
        print("Saved viral rate plot:", p)

    if "shares_log" in df.columns:
        plt.figure(figsize=(6, 4))
        df.groupby("cluster_kmeans")["shares_log"].mean().plot(kind="bar")
        plt.title("Avg Log Shares by Cluster")
        plt.tight_layout()
        p = fig_dir / "cluster_avg_log_shares.png"
        plt.savefig(p)
        plt.close()
        print("Saved log shares plot:", p)


def main():
    sns.set_theme()

    project_root = Path(__file__).resolve().parents[1]
    data_path = find_features_complete(project_root)

    print("Loading:", data_path)
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # Select features
    X, feature_cols = select_cluster_features(df)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute metrics
    k_range = range(2, 11)
    silhouette_scores, inertia_values = compute_kmeans_metrics(X_scaled, k_range)

    # Choose best k by silhouette
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print("\nBest k =", best_k)

    # Fit final kmeans
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster_kmeans"] = km.fit_predict(X_scaled)

    # Output dirs
    processed_dir = project_root / "data" / "processed"
    fig_dir = project_root / "figures" / "step6_clustering"
    processed_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    df.to_csv(processed_dir / "features_with_clusters.csv", index=False)
    print("\nSaved updated dataset with clusters.")

    # Save metrics
    pd.DataFrame({
        "k": list(k_range),
        "silhouette": [silhouette_scores[k] for k in k_range],
        "inertia": [inertia_values[k] for k in k_range]
    }).to_csv(processed_dir / "cluster_metrics.csv", index=False)

    # Plotting
    plot_silhouette(silhouette_scores, fig_dir / "silhouette_scores.png")
    plot_elbow(inertia_values, fig_dir / "elbow_plot.png")
    plot_pca_clusters(X_scaled, df["cluster_kmeans"], fig_dir / "pca_clusters.png")
    plot_cluster_bars(df, fig_dir)

    print("\nStep 6 complete.")


if __name__ == "__main__":
    main()