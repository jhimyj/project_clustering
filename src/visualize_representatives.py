import os
import matplotlib.pyplot as plt
from PIL import Image

def show_multiple_per_cluster(metadata, posters_dir, max_per_cluster=5, image_size=(120, 180)):
    "Muestra varias películas por cluster para evaluar visualmente su coherencia."
    clusters = sorted(metadata['cluster'].unique())

    for cluster_id in clusters:
        cluster_data = metadata[metadata['cluster'] == cluster_id]
        sample = cluster_data.sample(min(len(cluster_data), max_per_cluster), random_state=42)

        fig, axes = plt.subplots(1, len(sample), figsize=(len(sample)*2, 3))
        fig.suptitle(f"Cluster {cluster_id} - {len(cluster_data)} películas", fontsize=14)

        if len(sample) == 1:
            axes = [axes]

        for ax, (_, row) in zip(axes, sample.iterrows()):
            movie_id = row["movieId"]
            img_path = os.path.join(posters_dir, f"{movie_id}.jpg")

            if os.path.exists(img_path):
                img = Image.open(img_path).resize(image_size)
                ax.imshow(img)
                ax.set_title(f"{movie_id}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No image", ha="center", va="center")
            ax.axis("off")

        plt.tight_layout()
        plt.show()


def show_recommendations(movie_id, recommended_ids, posters_dir, image_size=(120, 180)):
    " Muestra el póster de la película original + sus recomendaciones"
    all_ids = [movie_id] + [int(mid) for mid in recommended_ids]

    fig, axes = plt.subplots(1, len(all_ids), figsize=(len(all_ids)*2, 3))
    fig.suptitle(f"Recomendaciones para movieId {movie_id}", fontsize=14)

    if len(all_ids) == 1:
        axes = [axes]

    for ax, mid in zip(axes, all_ids):
        img_path = os.path.join(posters_dir, f"{mid}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).resize(image_size)
            ax.imshow(img)
            ax.set_title(f"ID: {mid}", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No image", ha='center', va='center')
        ax.axis("off")

    plt.tight_layout()
    plt.show()