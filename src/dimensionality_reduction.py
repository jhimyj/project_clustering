import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def apply_pca(features: np.ndarray, var_threshold: float = 0.92) -> np.ndarray:
    pca = PCA(n_components=var_threshold)
    reduced = pca.fit_transform(features)
    print(f"[PCA] Componentes: {pca.n_components_}, Varianza explicada: {np.sum(pca.explained_variance_ratio_):.4f}")
    return reduced

def apply_svd(features: np.ndarray, var_threshold: float = 0.92) -> np.ndarray:
    # SVD no soporta n_components como proporción → debemos encontrarlo manualmente
    temp_svd = TruncatedSVD(n_components=min(features.shape[1] - 1, 1000))
    temp_svd.fit(features)
    explained = np.cumsum(temp_svd.explained_variance_ratio_)

    n_components = np.searchsorted(explained, var_threshold) + 1
    svd = TruncatedSVD(n_components=n_components)
    reduced = svd.fit_transform(features)
    print(f"[SVD] Componentes: {n_components}, Varianza explicada: {explained[n_components - 1]:.4f}")
    return reduced
