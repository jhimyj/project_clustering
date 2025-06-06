import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from utils import process_metadata_for_year_column
from src.load_data import load_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

FEATURES_PATH = "features.npy"
METADATA_PATH = "src/movies_with_posters.csv"
TSNE_COORDS_PATH = "features_tsne.npy"
POSTER_DIR = "data/posters"
ID_COLUMN_IN_METADATA = "movieId"

LABEL_FILES = {
    "KMeans (PCA)": "labels_kmeans_PCA.npy",
    "KMeans (SVD)": "labels_kmeans_SVD.npy",
    "Agglomerative (PCA)": "labels_agglomerative_PCA.npy",
    "Agglomerative (SVD)": "labels_agglomerative_SVD.npy"
}

def get_cluster_representatives(metadata_df_cluster, features_cluster, num_representatives=5):
    "Selecciona películas representativas de un clúster "
    if metadata_df_cluster.empty:
        return pd.DataFrame()
    
    if len(metadata_df_cluster) <= num_representatives:
        return metadata_df_cluster
    
    # Si tienes las características del clúster (features_cluster):
    if features_cluster is not None and len(features_cluster) > 0:
        try:
            centroid = np.mean(features_cluster, axis=0)
            distances = np.linalg.norm(features_cluster - centroid, axis=1) # Distancia Euclidiana
            closest_indices = np.argsort(distances)[:num_representatives] # Para distancia (menor es mejor)
            return metadata_df_cluster.iloc[closest_indices]
        except Exception as e:
            st.warning(f"No se pudo calcular representantes por centroide (error: {e}), mostrando aleatorios.")
            return metadata_df_cluster.sample(min(num_representatives, len(metadata_df_cluster)))
    else: #si no hay features para el cluster
        return metadata_df_cluster.sample(min(num_representatives, len(metadata_df_cluster)))


def run(filters):
    st.header("🌐 Visualización de Clústeres de Películas")

    # --- 1. Selección del Algoritmo de Clustering ---
    st.subheader("1. Selección de Etiquetas de Clúster")
    selected_algorithm_display_name = st.radio(
        "Elige el conjunto de etiquetas de clúster a visualizar:",
        options=list(LABEL_FILES.keys()),
        horizontal=True,
        key="cluster_algo_selection"
    )
    chosen_label_file = LABEL_FILES[selected_algorithm_display_name]
    st.caption(f"Usando etiquetas de: '{chosen_label_file}'")

    try:
        features_all, initial_metadata_from_load = load_data(FEATURES_PATH, METADATA_PATH, chosen_label_file)
        
        if initial_metadata_from_load is None:
            st.error("No se pudieron cargar los metadatos con las etiquetas del clúster.")
            return
        if features_all is None:
            st.warning("No se pudieron cargar las características. Algunas visualizaciones podrían no estar disponibles.")

        metadata_df = process_metadata_for_year_column(initial_metadata_from_load.copy())
        
        if ID_COLUMN_IN_METADATA not in metadata_df.columns:
            st.error(f"Error: Columna ID '{ID_COLUMN_IN_METADATA}' no encontrada en metadatos.")
            return
        metadata_df[ID_COLUMN_IN_METADATA] = metadata_df[ID_COLUMN_IN_METADATA].astype(str)

        if 'cluster' not in metadata_df.columns:
            st.error(f"Error: La columna 'cluster' no fue encontrada en los metadatos después de cargar")
            return

    except Exception as e:
        st.error(f"Error crítico al cargar datos para visualización de clústeres: {e}")
        import traceback
        st.error(traceback.format_exc())
        return
    
    st.markdown("---")

    filtered_df = metadata_df.copy()
    original_indices = np.arange(len(features_all)) if features_all is not None else np.array([])

    if filters.get('genre'):
        active_genre_filters = filters['genre']
        def check_movie_genres(genres_val):
            if pd.isna(genres_val): return False
            movie_genre_list = []
            if isinstance(genres_val, str): movie_genre_list = [g.strip() for g in genres_val.split('|') if g.strip()]
            elif isinstance(genres_val, list): movie_genre_list = [str(g).strip() for g in genres_val if str(g).strip()]
            return any(g_filter in movie_genre_list for g_filter in active_genre_filters)
        if 'genres' in filtered_df.columns:
            genre_mask = filtered_df['genres'].apply(check_movie_genres)
            filtered_df = filtered_df[genre_mask]
            if features_all is not None: original_indices = original_indices[genre_mask.values]


    if filters.get('year_range') and 'year' in filtered_df.columns:
        try:
            valid_years_mask = pd.notna(filtered_df['year'])
            temp_df_with_valid_years = filtered_df[valid_years_mask]
            
            if not temp_df_with_valid_years.empty:
                year_min_filter, year_max_filter = filters['year_range']
                year_comparison_mask = (temp_df_with_valid_years['year'].astype(int) >= year_min_filter) & \
                                       (temp_df_with_valid_years['year'].astype(int) <= year_max_filter)
                
                final_year_mask = pd.Series(False, index=filtered_df.index)
                final_year_mask.loc[temp_df_with_valid_years[year_comparison_mask].index] = True
                
                filtered_df = filtered_df[final_year_mask]
                if features_all is not None: original_indices = original_indices[final_year_mask.values[original_indices]] # Asegurar que el índice es correcto

        except Exception as e:
            st.warning(f"Advertencia: No se pudo aplicar el filtro de año (error: {e}).")

    if filtered_df.empty:
        st.warning("No hay películas que coincidan con los filtros seleccionados.")
        return

    # Filtrar las características si existen, usando los índices de las filas que quedaron en filtered_df
    features_filtered = None
    if features_all is not None and len(original_indices) > 0 and len(original_indices) <= len(features_all):
        try:
            features_filtered = features_all[filtered_df.index.tolist()] # Esto asume que filtered_df.index son los índices originales
        except IndexError: # Si el índice de filtered_df fue reseteado y ya no alinea con features_all
             st.warning("No se pudieron alinear las características filtradas con los metadatos filtrados debido a reseteo de índice. Los representantes de clúster podrían ser aleatorios.")
    

    st.markdown("---")
    
    # --- 4. Mostrar Películas Representativas ---
    st.subheader(f"2. Películas Representativas por Clúster ({selected_algorithm_display_name})")
    num_representatives = st.slider("Número de representantes por clúster:", 1, 10, 3, key="num_reps_slider")
    
    unique_clusters = sorted(filtered_df['cluster'].unique())

    for cluster_id in unique_clusters:
        st.markdown(f"#### Clúster {cluster_id}")
        movies_in_cluster_df = filtered_df[filtered_df['cluster'] == cluster_id]
        
        features_current_cluster = None
        if features_filtered is not None:
            try:
                indices_for_features = movies_in_cluster_df.index.tolist()
                features_current_cluster = features_all[indices_for_features]
            except Exception as e:
                st.warning(f"No se pudieron obtener features para el clúster {cluster_id}. Representantes serán aleatorios.")


        representatives_df = get_cluster_representatives(movies_in_cluster_df, features_current_cluster, num_representatives)
        
        if representatives_df.empty:
            st.write("No hay películas en este clúster después de aplicar filtros.")
            continue

        cols = st.columns(num_representatives)
        for i, (_, movie_row) in enumerate(representatives_df.iterrows()):
            with cols[i % num_representatives]:
                title_html = f"<p style='font-weight: bold; min-height: 3em; margin-bottom: 0.1em;'>{movie_row.get('title', 'N/A')}</p>"
                st.markdown(title_html, unsafe_allow_html=True)
                
                poster_path = os.path.join(POSTER_DIR, f"{movie_row[ID_COLUMN_IN_METADATA]}.jpg")
                if os.path.exists(poster_path):
                    st.image(poster_path, use_container_width=True)
                else:
                    st.caption("Póster no disponible")
                
                genres_text = ", ".join(movie_row.get('genres', ["N/A"])) if isinstance(movie_row.get('genres'), list) else movie_row.get('genres', "N/A")
                st.caption(f"Géneros: {genres_text}")
                if pd.notna(movie_row.get('year')):
                    st.caption(f"Año: {int(movie_row['year'])}")
        st.markdown("---")


    # --- 5. Visualización 2D de la Distribución ---
    st.subheader(f"3. Distribución 2D de Películas ({selected_algorithm_display_name})")
    
    coords_2d = None
    # Cargar coordenadas precalculadas
    if os.path.exists(TSNE_COORDS_PATH):
        try:
            all_coords_2d = np.load(TSNE_COORDS_PATH)
            if len(all_coords_2d) == len(metadata_df): # Comprobar si coincide con los metadatos originales
                coords_2d = all_coords_2d[filtered_df.index.tolist()]
            else:
                st.warning(f"El archivo '{TSNE_COORDS_PATH}' no coincide en longitud con los metadatos. No se usará.")
        except Exception as e:
            st.warning(f"No se pudo cargar '{TSNE_COORDS_PATH}': {e}")
    if coords_2d is not None and len(coords_2d) == len(filtered_df):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Colorear por clúster
        scatter_labels = filtered_df['cluster'].values
        unique_scatter_labels = np.unique(scatter_labels)
        
        # Usar un colormap discreto
        cmap = plt.get_cmap('viridis', len(unique_scatter_labels))

        for i, cluster_val in enumerate(unique_scatter_labels):
            cluster_mask = (scatter_labels == cluster_val)
            ax.scatter(coords_2d[cluster_mask, 0], coords_2d[cluster_mask, 1], 
                       color=cmap(i), label=f'Clúster {cluster_val}', alpha=0.7, s=30)

        ax.set_title(f"Distribución 2D de Películas (Coloreado por Clúster - {selected_algorithm_display_name})")
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        if len(unique_scatter_labels) <= 15: # No mostrar leyenda si hay demasiados clusters
            ax.legend()
        st.pyplot(fig)
    elif coords_2d is not None and len(coords_2d) != len(filtered_df) :
        st.warning(f"Desajuste en el número de coordenadas 2D ({len(coords_2d)}) y películas filtradas ({len(filtered_df)}). No se puede graficar.")
    else:
        st.info("No hay datos 2D para mostrar con la configuración actual.")