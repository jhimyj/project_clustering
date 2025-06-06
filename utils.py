import streamlit as st
import os
import pandas as pd
import numpy as np
import re
from src.load_data import load_data
from src.recommendation import recommend_similar

POSTER_DIR = "data/posters"
ID_COLUMN_IN_METADATA = "movieId"

def extract_year_from_title(title_str):
    if pd.isna(title_str):
        return pd.NA
    match = re.search(r'\((\d{4})\)$', str(title_str))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return pd.NA
    return pd.NA

def process_metadata_for_year_column(metadata_df: pd.DataFrame) -> pd.DataFrame:
    if 'title' in metadata_df.columns:
        metadata_df['year'] = metadata_df['title'].apply(extract_year_from_title)
        metadata_df['year'] = metadata_df['year'].astype('Int64')
    else:
        st.warning("Advertencia (desde utils.py): La columna 'title' no se encontró. No se pudo extraer el año.")
        metadata_df['year'] = pd.Series([pd.NA] * len(metadata_df), dtype='Int64')
    return metadata_df

def recommend_similar_by_id(movie_id, metadata_path: str, features_path: str, label_file_name: str, filters: dict, top_k: int = 10):
    try:
        features, metadata_from_load = load_data(features_path, metadata_path, label_file_name)

        if features is None or metadata_from_load is None:
            st.error("Error (desde utils.py): No se pudieron cargar los datos.")
            return []
        
        metadata = process_metadata_for_year_column(metadata_from_load.copy())

        if ID_COLUMN_IN_METADATA not in metadata.columns:
            st.error(f"Error crítico (utils.py): Columna ID '{ID_COLUMN_IN_METADATA}' no encontrada. Verifica la constante.")
            return []
        
        raw_recommendations = recommend_similar(movie_id, features, metadata, top_k=top_k)

        potential_recommendation_metadata_rows = []
        # obtener las filas de metadatos de las recomendaciones candidatas
        if isinstance(raw_recommendations, list) and all(isinstance(x, (int, str, np.integer, np.str_)) for x in raw_recommendations):
            for rec_id_candidate in raw_recommendations:
                row_df = metadata[metadata[ID_COLUMN_IN_METADATA].astype(str) == str(rec_id_candidate)] # Comparación como string por seguridad
                if not row_df.empty:
                    potential_recommendation_metadata_rows.append(row_df.iloc[0])
        elif isinstance(raw_recommendations, pd.DataFrame):
            id_col_in_recs_df = 'id' if 'id' in raw_recommendations.columns else ID_COLUMN_IN_METADATA
            for _, rec_row_from_df in raw_recommendations.iterrows():
                rec_id_candidate = rec_row_from_df[id_col_in_recs_df]
                full_metadata_row_df = metadata[metadata[ID_COLUMN_IN_METADATA].astype(str) == str(rec_id_candidate)] # Comparación como string
                if not full_metadata_row_df.empty:
                    potential_recommendation_metadata_rows.append(full_metadata_row_df.iloc[0])
        else:
            st.error(f"Formato inesperado de recomendaciones iniciales: {type(raw_recommendations)}.")
            return []

        filtered_recommendation_results = []
        for rec_movie_meta_row in potential_recommendation_metadata_rows:
            genre_match = True
            if filters.get('genre'):
                movie_genres_val = rec_movie_meta_row.get('genres', [])
                current_movie_genre_list = []
                if isinstance(movie_genres_val, str):
                    current_movie_genre_list = [g.strip() for g in movie_genres_val.split('|') if g.strip()]
                elif isinstance(movie_genres_val, list):
                    current_movie_genre_list = [str(g).strip() for g in movie_genres_val if str(g).strip()]
                if not current_movie_genre_list or not any(g_filter in current_movie_genre_list for g_filter in filters['genre']):
                    genre_match = False
            
            year_match = True
            if genre_match and year_match:
                rec_id = rec_movie_meta_row[ID_COLUMN_IN_METADATA]
                # Obtener géneros para mostrar
                genres_for_display_val = rec_movie_meta_row.get('genres', [])
                genres_text_for_display = "Géneros no disponibles"
                if isinstance(genres_for_display_val, str):
                    genres_text_for_display = ", ".join([g.strip() for g in genres_for_display_val.split('|') if g.strip()])
                elif isinstance(genres_for_display_val, list):
                    genres_text_for_display = ", ".join([str(g).strip() for g in genres_for_display_val if str(g).strip()])
                if not genres_text_for_display: genres_text_for_display = "Sin especificar"


                filtered_recommendation_results.append({
                    "title": rec_movie_meta_row.get("title", f"ID: {rec_id}"),
                    "poster_path": os.path.join(POSTER_DIR, f"{rec_id}.jpg"),
                    "genres": genres_text_for_display # Añadir géneros al resultado
                })
        return filtered_recommendation_results

    except FileNotFoundError as e:
        st.error(f"Error en utils.py (Archivo no encontrado): {e}")
        return []
    except KeyError as e:
        st.error(f"Error en utils.py (KeyError, posible columna faltante como '{ID_COLUMN_IN_METADATA}' o 'title', o ID no encontrado): {e}")
        return []
    except Exception as e:
        st.error(f"Error general en utils.py: {e}")
        import traceback
        st.error(traceback.format_exc())
        return []