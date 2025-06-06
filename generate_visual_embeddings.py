import numpy as np
from sklearn.manifold import TSNE
import time
import os

INPUT_FEATURES_FILE = "features.npy"
OUTPUT_TSNE_FILE = "features_tsne.npy"

# --- Parámetros de t-SNE ---
N_COMPONENTS = 2       # Número de dimensiones a reducir (2 para visualización 2D).
RANDOM_STATE = 42      # Semilla
PERPLEXITY = 30.0      # Numero de vecinos
N_ITER = 1000          # Número de iteraciones
LEARNING_RATE = 'auto'
METRIC = 'cosine'

def calcular_y_guardar_tsne():
    " Carga las características, calcula t-SNE y guarda las coordenadas 2D"
    print(f"Iniciando el cálculo de t-SNE...")

    # Cargar las características originales
    if not os.path.exists(INPUT_FEATURES_FILE):
        print(f"Error: El archivo de características '{INPUT_FEATURES_FILE}' no fue encontrado.")
        return

    try:
        features_high_dim = np.load(INPUT_FEATURES_FILE)
        print(f"Características cargadas. Forma: {features_high_dim.shape}")
        if features_high_dim.ndim != 2:
            print(f"Error: Se esperaba un array 2D de características, pero se obtuvo forma {features_high_dim.shape}.")
            return
        if features_high_dim.shape[0] == 0:
            print(f"Error: El archivo de características está vacío.")
            return

    except Exception as e:
        print(f"Error al cargar el archivo de características: {e}")
        return

    tsne_model = TSNE(
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE,
        perplexity=PERPLEXITY,
        n_iter=N_ITER,
        learning_rate=LEARNING_RATE,
        metric=METRIC,
        init='pca'
    )

    try:
        features_2d = tsne_model.fit_transform(features_high_dim)
    except Exception as e:
        print(f"Error durante el cálculo de t-SNE: {e}")
    # Guardar las características 2D resultantes
    try:
        print(f"\nGuardando las características t-SNE en '{OUTPUT_TSNE_FILE}'...")
        np.save(OUTPUT_TSNE_FILE, features_2d)
        print(f"¡Éxito! Características t-SNE guardadas en '{os.path.abspath(OUTPUT_TSNE_FILE)}'.")
    except Exception as e:
        print(f"Error al guardar el archivo t-SNE: {e}")

if __name__ == "__main__":
    calcular_y_guardar_tsne()