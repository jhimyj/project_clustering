import streamlit as st
import os
import pandas as pd
from utils import recommend_similar_by_id, POSTER_DIR, process_metadata_for_year_column
from src.load_data import load_data

FEATURES_PATH = "features.npy"
METADATA_PATH = "src/movies_with_posters.csv"
ID_COLUMN_IN_METADATA = "movieId"

LABEL_FILES = {
    "KMeans (PCA)": "labels_kmeans_PCA.npy",
    "KMeans (SVD)": "labels_kmeans_SVD.npy",
    "Agglomerative (PCA)": "labels_agglomerative_PCA.npy",
    "Agglomerative (SVD)": "labels_agglomerative_SVD.npy"
}

def run(filters):
    st.header("🔍 Buscar películas similares desde un póster existente")

    # Carga y procesamiento inicial de metadatos
    try:
        _, initial_metadata_from_load = load_data(FEATURES_PATH, METADATA_PATH, labels_path=None)
        if initial_metadata_from_load is None:
            st.error("No se pudo cargar la info inicial de películas para filtros.")
            return
        initial_metadata_df = process_metadata_for_year_column(initial_metadata_from_load.copy())
        if ID_COLUMN_IN_METADATA not in initial_metadata_df.columns:
            st.error(f"Error (search_by_image.py): Columna ID '{ID_COLUMN_IN_METADATA}' no encontrada.")
            return
        initial_metadata_df[ID_COLUMN_IN_METADATA] = initial_metadata_df[ID_COLUMN_IN_METADATA].astype(str)
    except Exception as e:
        st.error(f"Error crítico al cargar/procesar metadatos iniciales: {e}")
        return

    # Configuración del Algoritmo de Clustering
    st.subheader("1. Configuración del Algoritmo de Clustering")
    selected_algorithm_display_name = st.radio(
        "Selecciona el conjunto de etiquetas de cluster a utilizar:",
        options=list(LABEL_FILES.keys()),
        horizontal=True
    )
    chosen_label_file = LABEL_FILES[selected_algorithm_display_name]
    st.caption(f"Se usarán las etiquetas de: '{chosen_label_file}'")
    st.markdown("---")

    # Selección de Película de Referencia (con filtros y selección persistente)
    st.subheader("2. Selección de Película de Referencia")

    all_poster_movie_ids_from_filenames = []
    if os.path.isdir(POSTER_DIR):
        all_poster_movie_ids_from_filenames = sorted([f.replace(".jpg", "") for f in os.listdir(POSTER_DIR) if f.endswith(".jpg")])
    else:
        st.error(f"El directorio de pósters '{POSTER_DIR}' no fue encontrado. Verifica la ruta.")
        return
    if not all_poster_movie_ids_from_filenames:
        st.warning(f"No se encontraron pósters (.jpg) en '{POSTER_DIR}'.")
        return

    filtered_selectable_df = initial_metadata_df.copy()
    # Filtro de género
    if filters.get('genre'):
        active_genre_filters = filters['genre']
        def check_movie_genres(genres_val):
            if pd.isna(genres_val): return False
            movie_genre_list = []
            if isinstance(genres_val, str): movie_genre_list = [g.strip() for g in genres_val.split('|') if g.strip()]
            elif isinstance(genres_val, list): movie_genre_list = [str(g).strip() for g in genres_val if str(g).strip()]
            return any(g_filter in movie_genre_list for g_filter in active_genre_filters)
        if 'genres' in filtered_selectable_df.columns:
            filtered_selectable_df = filtered_selectable_df[filtered_selectable_df['genres'].apply(check_movie_genres)]

    # Filtro de año
    if filters.get('year_range') and 'year' in filtered_selectable_df.columns:
        try:
            valid_years_mask = pd.notna(filtered_selectable_df['year'])
            temp_df_with_valid_years = filtered_selectable_df[valid_years_mask]
            if not temp_df_with_valid_years.empty:
                 year_min_filter, year_max_filter = filters['year_range']
                 year_comparison_mask = (temp_df_with_valid_years['year'].astype(int) >= year_min_filter) & \
                                        (temp_df_with_valid_years['year'].astype(int) <= year_max_filter)
                 filtered_ids_by_year = temp_df_with_valid_years[year_comparison_mask][ID_COLUMN_IN_METADATA]
                 filtered_selectable_df = filtered_selectable_df[filtered_selectable_df[ID_COLUMN_IN_METADATA].isin(filtered_ids_by_year)]
            else:
                 filtered_selectable_df = pd.DataFrame(columns=filtered_selectable_df.columns)
        except Exception as e:
            st.warning(f"Advertencia: No se pudo aplicar el filtro de año a la lista de selección (error: {e}).")


    filtered_movie_ids_from_metadata = filtered_selectable_df[ID_COLUMN_IN_METADATA].astype(str).unique().tolist()
    selectable_movie_ids_for_dropdown = sorted(list(set(all_poster_movie_ids_from_filenames) & set(filtered_movie_ids_from_metadata)))

    # --- Lógica para mantener la selección del selectbox usando st.session_state ---
    if not selectable_movie_ids_for_dropdown:
        st.warning("No hay películas que coincidan con los filtros y que tengan un póster disponible para seleccionar.")
        # Limpiar la selección si no hay opciones
        if "movie_reference_selection" in st.session_state:
            del st.session_state.movie_reference_selection
        selected_movie_id_str = None
    else:
        # Si la selección previa (guardada en session_state) ya no es válida con los nuevos filtros,
        if "movie_reference_selection" not in st.session_state or \
           st.session_state.movie_reference_selection not in selectable_movie_ids_for_dropdown:
            st.session_state.movie_reference_selection = selectable_movie_ids_for_dropdown[0]

        try:
            current_selection_index = selectable_movie_ids_for_dropdown.index(st.session_state.movie_reference_selection)
        except ValueError: # No debería ocurrir si la lógica anterior es correcta
            current_selection_index = 0 
        
        selected_movie_id_str = st.selectbox(
            f"Selecciona un póster por su ID ({len(selectable_movie_ids_for_dropdown)} películas coinciden con filtros):",
            options=selectable_movie_ids_for_dropdown,
            index=current_selection_index,
            key="movie_reference_selection" # Usar esta key para que Streamlit maneje el estado
        )

    if selected_movie_id_str: # Procede solo si hay una película seleccionada
        try:
            query_movie_id_for_recs = int(selected_movie_id_str)
        except ValueError:
            query_movie_id_for_recs = selected_movie_id_str
        
        image_path = os.path.join(POSTER_DIR, f"{selected_movie_id_str}.jpg")
        if os.path.exists(image_path):
            st.image(image_path, caption=f"Póster Seleccionado: ID {selected_movie_id_str}", width=200)

            if st.button(f"Buscar películas similares a ID: {selected_movie_id_str}", type="primary"):
                with st.spinner("Buscando recomendaciones y aplicando filtros..."):
                    recommended_movies_list = recommend_similar_by_id(
                        movie_id=query_movie_id_for_recs,
                        metadata_path=METADATA_PATH, features_path=FEATURES_PATH,
                        label_file_name=chosen_label_file, filters=filters, top_k=30
                    )
                
                if recommended_movies_list:
                    st.subheader(f"🎯 Películas Similares ({len(recommended_movies_list)} encontradas post-filtros):")
                    num_display_cols = 5
                    max_recs_to_show = 10                    
                    cols = st.columns(num_display_cols)
                    for i, rec_movie in enumerate(recommended_movies_list):
                        if i >= max_recs_to_show:
                            st.caption(f"... y {len(recommended_movies_list) - max_recs_to_show} más (no mostradas).")
                            break
                        with cols[i % num_display_cols]:
                            # Usar st.markdown para más control sobre el formato del título
                            title_html = f"<p style='font-weight: bold; min-height: 3em; margin-bottom: 0.1em;'>{rec_movie.get('title', 'N/A')}</p>"
                            st.markdown(title_html, unsafe_allow_html=True)
                            
                            if os.path.exists(rec_movie["poster_path"]):
                                st.image(rec_movie["poster_path"], use_container_width=True)
                            else:
                                st.warning(f"Póster no hallado: {rec_movie['poster_path']}")
                            
                            # Mostrar géneros debajo del póster
                            st.caption(f"Géneros: {rec_movie.get('genres', 'N/A')}")
                else:
                    st.info("No se encontraron recomendaciones que coincidan con los filtros aplicados.")
        else:
            st.error(f"El archivo de póster no fue encontrado: {image_path}")