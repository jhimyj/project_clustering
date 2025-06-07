import streamlit as st
from components import search_by_image, show_clusters, up_image
from components.filters import get_filters

st.set_page_config(page_title="Visualizador de Recomendaciones", layout="wide")

st.title("🎬 Visualizador del Sistema de Recomendación de Películas")

option = st.sidebar.selectbox("Selecciona una opción", [
    "🔍 Buscar por imagen",
    "🌐 Ver clústeres representativos",
    "📤 Subir foto y elegir clúster",
])

filters = get_filters()

if option == "🔍 Buscar por imagen":
    search_by_image.run(filters)

elif option == "🌐 Ver clústeres representativos":
    show_clusters.run(filters)
elif option == "📤 Subir foto y elegir clúster":
    up_image.run(filters)
    pass
