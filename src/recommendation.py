# src/recommendation.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cdist


def recommend_similar(movie_id, features, metadata, top_k=5):
    if movie_id not in metadata["movieId"].values:
        raise ValueError(f"movieId {movie_id} no encontrado")

    idx = metadata[metadata["movieId"] == movie_id].index[0]
    target_feature = features[idx].reshape(1, -1)
    target_cluster = metadata.loc[idx, "cluster"]

    # Obtener características del mismo cluster
    cluster_indices = metadata[metadata["cluster"] == target_cluster].index
    cluster_features = features[cluster_indices]

    # calcular distancias coseno
    distances = cosine_distances(target_feature, cluster_features).flatten()
    ranked_indices = cluster_indices[np.argsort(distances)]
    recommended_ids = metadata.loc[ranked_indices, "movieId"].values

    # Excluir la película de entrada
    recommended_ids = [mid for mid in recommended_ids if mid != movie_id][:top_k]
    return recommended_ids

