#!pip install streamlit
import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import pairwise_distances
from utils import process_metadata_for_year_column
from src.load_data import load_data
from src.feature_extraction import FeatureExtractor

FEATURES_PATH = "features.npy"
METADATA_PATH = "src/movies_with_posters.csv"
POSTER_DIR = "data/posters"
ID_COLUMN_IN_METADATA = "movieId"

LABEL_FILES = {
    "KMeans (PCA)": "labels_kmeans_PCA.npy",
    "KMeans (SVD)": "labels_kmeans_SVD.npy",
    "Agglomerative (PCA)": "labels_agglomerative_PCA.npy",
    "Agglomerative (SVD)": "labels_agglomerative_SVD.npy"
}


def run(filters):
    st.header("📤 Subir Imagen y Recomendar Películas")

    selected_algo = st.radio(
        "Selecciona conjunto de etiquetas (cluster):",
        options=list(LABEL_FILES.keys()),
        horizontal=True
    )
    label_file = LABEL_FILES[selected_algo]
    st.caption(f"Usando etiquetas de: {label_file}")
    st.markdown("---")

    uploaded = st.file_uploader("Sube un póster (.jpg, .png)", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Espera a que subas una imagen para continuar.")
        return


    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.getbuffer())
        image_path = tmp.name

    try:
        img = Image.open(image_path).convert("RGB")
        st.image(img, caption="Imagen cargada", use_container_width=True)
    except Exception as e:
        st.error(f"Error al cargar imagen: {e}")
        return

    with st.spinner("Extrayendo características…"):
        try:
            fe = FeatureExtractor(image_size=(224, 224), use_cache=False)
            features_raw = fe.process_single_image(image_path)
            if features_raw is None:
                raise ValueError("Features vacíos")
        except Exception as e:
            st.error(f"No se pudieron extraer features: {e}")
            return

    try:
        features_all = np.load(FEATURES_PATH)
        labels = np.load(os.path.join(os.path.dirname(FEATURES_PATH), label_file))
        unique_labels = np.unique(labels)
        centroids = np.array([
            features_all[labels == lab].mean(axis=0)
            for lab in unique_labels
        ])
    except Exception as e:
        st.error(f"Error al cargar features/labels: {e}")
        return

    #Asignación de clúster
    features_new = features_raw.reshape(1, -1)
    dists = pairwise_distances(features_new, centroids, metric="euclidean")
    idx = np.argmin(dists, axis=1)[0]
    cluster_pred = int(unique_labels[idx])
    st.success(f"Clúster asignado: {cluster_pred}")
    st.markdown("---")

    try:
        feats, metadata = load_data(FEATURES_PATH, METADATA_PATH, labels_path=None)
        df = process_metadata_for_year_column(metadata.copy())
        df[ID_COLUMN_IN_METADATA] = df[ID_COLUMN_IN_METADATA].astype(str)
        df['cluster'] = labels.astype(int)
    except Exception as e:
        st.error(f"Error cargando metadata/clusters: {e}")
        return


    if filters.get('genre') and 'genres' in df.columns:
        gens = filters['genre']
        df = df[df['genres'].apply(lambda g: any(x in (g or '').split('|') for x in gens))]

    if filters.get('year_range') and 'year' in df.columns:
        ymin, ymax = filters['year_range']
        df = df[pd.to_numeric(df['year'], errors='coerce').between(ymin, ymax)]

    #películas del mismo clúster
    df_cluster = df[df['cluster'] == cluster_pred]
    if df_cluster.empty:
        st.info("No se encontraron películas en este clúster con los filtros aplicados.")
        return


    idxs = df_cluster.index.tolist()
    feats_cluster = features_all[idxs]
    dists_new = pairwise_distances(features_new, feats_cluster).flatten()
    df_cluster = df_cluster.assign(distance=dists_new).sort_values('distance').head(10)

    st.subheader("Recomendaciones:")
    cols = st.columns(min(5, len(df_cluster)))
    for i, (_, row) in enumerate(df_cluster.iterrows()):
        with cols[i % 5]:
            st.markdown(f"**{row.get('title', 'N/A')}**")
            poster = os.path.join(POSTER_DIR, f"{row['movieId']}.jpg")
            if os.path.exists(poster):
                st.image(poster, use_container_width=True)
            else:
                st.caption("Póster no disponible")
            st.caption(f"Géneros: {row.get('genres','N/A')}")
            if pd.notna(row.get('year')):
                st.caption(f"Año: {int(float(row['year']))}")
