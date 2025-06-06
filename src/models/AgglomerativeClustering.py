import numpy as np

class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 10, linkage: str = 'average'):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters debe ser un entero positivo.")
        if linkage not in ['single', 'complete', 'average']:
            raise ValueError(f"Linkage debe ser 'single', 'complete', o 'average', no {linkage}.")

        self.num_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self._actual_clusters_indices = None 

    def precomputar_distancias(self, X: np.ndarray) -> np.ndarray:
        """Calcula las distancias euclidianas por pares iniciales entre puntos."""
        X_norm_sq = np.sum(X**2, axis=1)
        dist_sq = X_norm_sq[:, np.newaxis] + X_norm_sq[np.newaxis, :] - 2 * np.dot(X, X.T)
        dist_sq = np.maximum(dist_sq, 0)
        return np.sqrt(dist_sq)

    def fit(self, data: np.ndarray):
        n_samples = data.shape[0]

        if n_samples == 0:
            self.labels_ = np.array([])
            self._actual_clusters_indices = []
            return self
        n_cluster_max = max(1, min(self.num_clusters, n_samples))
        
        if n_samples <= n_cluster_max:
            self.labels_ = np.arange(n_samples)
            self._actual_clusters_indices = [[i] for i in range(n_samples)]
            return self

        dist_matrix = self.precomputar_distancias(data)
        np.fill_diagonal(dist_matrix, float('inf'))     # Evitar auto-distancias

        current_point_indices_in_cluster = [[i] for i in range(n_samples)]  # Lista de índices de puntos en cada clúster
        cluster_sizes = [1] * n_samples
        active_clusters_mask = [True] * n_samples
        num_current_clusters = n_samples
        
        for _ in range(n_samples - n_cluster_max):
            if num_current_clusters <= n_cluster_max:
                break

            min_dist_val = float('inf')
            merge_idx1, merge_idx2 = -1, -1 

            current_active_indices = [i for i, is_active in enumerate(active_clusters_mask) if is_active]

            if len(current_active_indices) < 2: break 
            
            for i_ptr in range(len(current_active_indices)):
                idx_i = current_active_indices[i_ptr]
                for j_ptr in range(i_ptr + 1, len(current_active_indices)):
                    idx_j = current_active_indices[j_ptr]
                    current_dist = dist_matrix[idx_i, idx_j]
                    if current_dist < min_dist_val:
                        min_dist_val = current_dist
                        merge_idx1, merge_idx2 = min(idx_i, idx_j), max(idx_i, idx_j)
            
            if merge_idx1 == -1: break
            
            current_point_indices_in_cluster[merge_idx1].extend(current_point_indices_in_cluster[merge_idx2])
            current_point_indices_in_cluster[merge_idx2] = [] 

            size1_before_merge = cluster_sizes[merge_idx1]
            size2_before_merge = cluster_sizes[merge_idx2]
            cluster_sizes[merge_idx1] = size1_before_merge + size2_before_merge
            cluster_sizes[merge_idx2] = 0 

            for k in range(n_samples): 
                if not active_clusters_mask[k] or k == merge_idx1 or k == merge_idx2:
                    continue

                dist_k_to_old1 = dist_matrix[k, merge_idx1] 
                dist_k_to_old2 = dist_matrix[k, merge_idx2]
                new_dist_k_to_merged: float

                if self.linkage == 'single':
                    new_dist_k_to_merged = min(dist_k_to_old1, dist_k_to_old2)
                elif self.linkage == 'complete':
                    new_dist_k_to_merged = max(dist_k_to_old1, dist_k_to_old2)
                elif self.linkage == 'average':
                    total_size = size1_before_merge + size2_before_merge
                    if total_size == 0: new_dist_k_to_merged = float('inf')
                    else:
                        new_dist_k_to_merged = (size1_before_merge * dist_k_to_old1 + 
                                                size2_before_merge * dist_k_to_old2) / total_size
                else:
                    raise ValueError(f"Criterio de enlace no soportado internamente: {self.linkage}")
                
                dist_matrix[k, merge_idx1] = new_dist_k_to_merged
                dist_matrix[merge_idx1, k] = new_dist_k_to_merged

            active_clusters_mask[merge_idx2] = False
            dist_matrix[merge_idx2, :] = float('inf')
            dist_matrix[:, merge_idx2] = float('inf')
            num_current_clusters -= 1

        self._actual_clusters_indices = [indices for i, indices in enumerate(current_point_indices_in_cluster) if active_clusters_mask[i]]
        
        final_labels = np.zeros(n_samples, dtype=int)
        for cluster_id, point_indices_list in enumerate(self._actual_clusters_indices):
            if not point_indices_list: continue 
            for point_idx in point_indices_list:
                final_labels[point_idx] = cluster_id
        self.labels_ = final_labels
        return self

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.labels_