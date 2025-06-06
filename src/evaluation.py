import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE

from collections import Counter, defaultdict
def evaluate_internal_metrics(features, labels):
    metrics = {}

    # Silhouette Score (cosine distance)
    silhouette = silhouette_score(features, labels, metric='cosine')
    metrics["silhouette_cosine"] = silhouette
    print(f"Silhouette Score (cosine): {silhouette:.4f}")

    # Silhouette Score (euclidean distance)
    silhouette_euc = silhouette_score(features, labels, metric='euclidean')
    metrics["silhouette_euclidean"] = silhouette_euc
    print(f"Silhouette Score (euclidean): {silhouette_euc:.4f}")

    # Davies-Bouldin Index
    db_index = davies_bouldin_score(features, labels)
    metrics["davies_bouldin"] = db_index
    print(f"Davies-Bouldin Index: {db_index:.4f}")

    # Calinski-Harabasz Index
    ch_index = calinski_harabasz_score(features, labels)
    metrics["calinski_harabasz"] = ch_index
    print(f"Calinski-Harabasz Index: {ch_index:.4f}")

    return metrics

def evaluate_genre_consistency(metadata, genre_column="genres"):
    cluster_genre_purity = {}
    for cluster_id in sorted(metadata["cluster"].unique()):
        cluster_data = metadata[metadata["cluster"] == cluster_id]
        genres = cluster_data[genre_column].dropna().astype(str).tolist()
        genre_counts = Counter()
        for g in genres:
            split_genres = [x.strip() for x in g.split('|')]
            genre_counts.update(split_genres)

        purity = genre_counts.most_common(1)[0][1] / len(genres) if genres else 0
        cluster_genre_purity[cluster_id] = {
            "purity": round(purity, 3),
            "top_genre": genre_counts.most_common(1)[0][0] if genre_counts else "None"
        }

    return cluster_genre_purity

def plot_tsne(features, labels, title="t-SNE"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, metric='cosine')
    tsne_result = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette='tab10', s=40)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_genre_distribution_heatmap(metadata, genre_column="genres", title ="distribución de géneros por cluster"):
    genre_matrix = defaultdict(Counter)
    for _, row in metadata.iterrows():
        cluster_id = row["cluster"]
        genres = str(row[genre_column]).split('|')
        for genre in genres:
            genre_matrix[cluster_id][genre.strip()] += 1

    df = pd.DataFrame(genre_matrix).fillna(0).astype(int).T
    plt.figure(figsize=(12, 6))
    sns.heatmap(df, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(title)
    plt.tight_layout()
    plt.show()
