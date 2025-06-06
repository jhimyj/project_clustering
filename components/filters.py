import streamlit as st

def get_filters():
    genre = st.sidebar.multiselect("Filtrar por género", options=[
        "Action", "Drama", "Comedy", "Sci-Fi", "Horror", "Romance", "Adventure"
    ])
    year = st.sidebar.slider("Año de lanzamiento", 1950, 2025, (2000, 2020))
    return {"genre": genre, "year_range": year}
