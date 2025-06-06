import numpy as np
from src.models.AgglomerativeClustering import AgglomerativeClustering
from src.models.KMeans import KMeans

def cluster_kmeans(data: np.ndarray, n_clusters: int = 10, type_reduction: str = "PCA") -> np.ndarray:
    kmeans = KMeans(num_cluster=n_clusters, random_state=42)
    labels = kmeans.fit(data)
    print(f"[KMeans] Clusters formados: {len(set(labels))}")
    np.save(f'labels_kmeans_{type_reduction}.npy', labels)  # Guardar etiquetas
    return labels

def cluster_agglomerative(data: np.ndarray, n_clusters: int = 10, type_reduction: str = "PCA") -> np.ndarray:
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agglomerative.fit_predict(data)
    print(f"[Agglomerative] Clusters formados: {len(set(labels))}")
    np.save(f'labels_agglomerative_{type_reduction}.npy', labels)  # Guardar etiquetas
    return labels

