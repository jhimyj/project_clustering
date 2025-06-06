import numpy as np
import random

class KMeans:
    def __init__(self, num_cluster, max_iter=1000, tol=1e-4, random_state=None):
        self.num_cluster = num_cluster      # Número de clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroides = None
        self.labels = None
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def distancia_euclidiana(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def inicializar_centroides(self, data):
        n_samples, n_features = data.shape
        centroids = np.zeros((self.num_cluster, n_features))    # Matriz de ceros
        centroids[0] = data[np.random.randint(n_samples)]       # Selecciona un punto aleatorio
        
        # Selecciona los siguientes centroides 
        for i in range(1, self.num_cluster):
            distances = np.min([np.linalg.norm(data - c, axis=1) for c in centroids[:i]], axis=0) # Distancia mínima 
            # Normaliza las distancias para obtener probabilidades y seleccionar el siguiente centroide
            probabilities = distances / np.sum(distances)
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()

            # Selecciona el siguiente centroide basado en las probabilidades
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids[i] = data[j]
                    break

        return centroids

    def asignar_clusters(self, data, centroides):
        clusters = {}
        labels = np.zeros(len(data))
        for i in range(len(data)):
            distancias = [self.distancia_euclidiana(data[i], c) for c in centroides]
            cluster = np.argmin(distancias)     # Devuelve el índice del menor valor
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(data[i])
            labels[i] = cluster
        labels = labels.astype(int)
        return clusters, labels
    def actualizar_centroides(self, centroides, clusters):
        new_centroides = np.zeros(centroides.shape)
        for i in range(self.num_cluster):
            new_centroides[i] = np.mean(clusters[i], axis=0)
        return new_centroides

    def fit(self, data):
        self.centroides = self.inicializar_centroides(data)
        for i in range(self.max_iter):
            clusters, self.labels = self.asignar_clusters(data, self.centroides)
            new_centroides = self.actualizar_centroides(self.centroides, clusters)
            if np.sum(np.abs(new_centroides - self.centroides)) < self.tol:
                #print("Convergió en la iteración", i)
                #print("centroides_last", self.centroides)
                #print("centroides_new", new_centroides)
                break
            self.centroides = new_centroides
        return self.labels

