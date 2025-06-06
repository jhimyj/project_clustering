import numpy as np
import pandas as pd
import cv2
import os
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from skimage.segmentation import slic
from skimage.measure import regionprops
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

class FeatureExtractor:
    def __init__(self, image_size=(224, 224), use_cache=True, n_jobs=-1):
        self.image_size = image_size
        self.use_cache = use_cache  # para optimizar el tiempo de procesamiento
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.feature_cache = {}
        
        # Configuración optimizada para diferentes tipos de características
        self.config = {
            'color_hist_bins': {'rgb': 32, 'hsv': [16, 16, 8], 'lab': [8, 8, 8]},
            'dominant_colors_k': 5,
            'lbp_radii': [1, 2],
            'glcm_angles': [0, 90],
            'gabor_params': [(0.1, 0), (0.3, 45), (0.5, 90)],
            'slic_segments': 50,
        }
        
        
    def extract_color_histograms(self, image):
        "Extrae histogramas de color RGB, HSV y LAB"
        
        cfg = self.config['color_hist_bins']

        # RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist_r = cv2.calcHist([rgb_image], [0], None, [cfg['rgb']], [0, 256]).flatten()
        hist_g = cv2.calcHist([rgb_image], [1], None, [cfg['rgb']], [0, 256]).flatten()
        hist_b = cv2.calcHist([rgb_image], [2], None, [cfg['rgb']], [0, 256]).flatten()

        # HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_image], [0], None, [cfg['hsv'][0]], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv_image], [1], None, [cfg['hsv'][1]], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv_image], [2], None, [cfg['hsv'][2]], [0, 256]).flatten()

        # LAB (mejor para percepción humana)
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hist_l = cv2.calcHist([lab_image], [0], None, [cfg['lab'][0]], [0, 256]).flatten()
        hist_a = cv2.calcHist([lab_image], [1], None, [cfg['lab'][1]], [0, 256]).flatten()
        hist_b_lab = cv2.calcHist([lab_image], [2], None, [cfg['lab'][2]], [0, 256]).flatten()

        # Normalizar
        histograms = [hist_r, hist_g, hist_b, hist_h, hist_s, hist_v, hist_l, hist_a, hist_b_lab]
        normalized_hists = []
        for hist in histograms:
            normalized_hists.append(hist / (np.sum(hist) + 1e-10))
        
        return np.concatenate(normalized_hists)

    def extract_dominant_colors(self, image, k=None):
        "Extrae características de colores dominantes"
        
        if k is None:
            k = self.config['dominant_colors_k']
        
        # Submuestrear la imagen para K-means más rápido
        small_image = cv2.resize(image, (64, 64))
        data = small_image.reshape((-1, 3)).astype(np.float32)
        
        # K-means más rápido
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        
        # Calcular porcentajes
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels)
        
        # Rellenar con ceros si hay menos de k colores
        color_features = []
        for i in range(k):
            if i < len(centers):
                color_features.extend(centers[i])
                color_features.append(percentages[i] if i < len(percentages) else 0)
            else:
                color_features.extend([0, 0, 0, 0])
        
        return np.array(color_features)
    
    def extract_composition_features(self, image):
        "Extrae características de composición visual"
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # División en tercios (regla de los tercios)
        third_h, third_w = h // 3, w // 3
        regions = []
        
        for i in range(3):
            for j in range(3):
                region = gray[i*third_h:(i+1)*third_h, j*third_w:(j+1)*third_w]
                regions.append([
                    np.mean(region),
                    np.std(region),
                    np.sum(region > np.mean(region)) / region.size  # Proporción de píxeles brillantes
                ])
        
        # Simetría horizontal y vertical
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)
        h_symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        
        top_half = gray[:h//2, :]
        bottom_half = cv2.flip(gray[h//2:, :], 0)
        v_symmetry = np.corrcoef(top_half.flatten(), bottom_half.flatten())[0, 1]
        
        # Contraste global
        global_contrast = np.std(gray)
        
        # Características de brillo
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        
        composition_features = np.array(regions).flatten().tolist()
        composition_features.extend([
            h_symmetry if not np.isnan(h_symmetry) else 0,
            v_symmetry if not np.isnan(v_symmetry) else 0,
            global_contrast,
            brightness_mean,
            brightness_std
        ])
        
        return np.array(composition_features)
    def extract_hu_moments(self, image):
        "Extrae momentos de Hu para características de forma"
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        central_moments = [
            moments['mu20'], moments['mu11'], moments['mu02'],
            moments['mu30'], moments['mu21'], moments['mu12'], moments['mu03']
        ]
        
        return np.concatenate([hu_moments, central_moments])  
    def extract_color_moments(self, image):
        "Extrae momentos de color"
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        color_moments = []
        
        for img, channels in [(rgb, 3), (hsv, 3)]:
            for c in range(channels):
                channel = img[:, :, c].astype(float)
                
                mean = np.mean(channel)
                var = np.var(channel)
                skew = np.mean(((channel - mean) / (np.sqrt(var) + 1e-10)) ** 3)
                
                color_moments.extend([mean, var, skew])
        
        return np.array(color_moments)
    def extract_texture(self, image):
        "Extrae características de textura"
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = []
        
        # LBP 
        for radius in self.config['lbp_radii']:
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            hist_lbp, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist_lbp = hist_lbp.astype(float) / (np.sum(hist_lbp) + 1e-10)
            features.extend(hist_lbp)
        
        # GLCM
        if gray.max() > 0:
            gray_reduced = (gray / 64).astype(np.uint8)  # Menos niveles de gris
            
            for angle in self.config['glcm_angles']:
                glcm = graycomatrix(gray_reduced, [1], [np.radians(angle)], 
                                 levels=4, symmetric=True, normed=True)  # Menos niveles
                
                props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
                for prop in props:
                    value = graycoprops(glcm, prop)[0, 0]
                    features.append(value)
        else:
            features.extend([0] * (len(self.config['glcm_angles']) * 5))
        
        # Gabor
        for frequency, theta in self.config['gabor_params']:
            try:
                real, _ = gabor(gray, frequency=frequency, theta=np.radians(theta))
                features.extend([np.mean(real), np.std(real), np.var(real)])
            except:
                features.extend([0, 0, 0])
        
        return np.array(features)

    def extract_statistical_features(self, image):
        "Características estadísticas rápidas y eficientes"
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Momentos estadísticos básicos
        features = [
            np.mean(gray),
            np.std(gray),
            np.var(gray),
            np.min(gray),
            np.max(gray),
            np.median(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75),
        ]
        
        # Características de distribución
        hist, _ = np.histogram(gray, bins=32, range=(0, 256))
        hist = hist / (np.sum(hist) + 1e-10)
        
        # Entropía
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        features.append(entropy)
        
        # Skewness y kurtosis aproximados
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        if std_val > 0:
            skewness = np.mean(((gray - mean_val) / std_val) ** 3)
            kurtosis = np.mean(((gray - mean_val) / std_val) ** 4) - 3
        else:
            skewness = kurtosis = 0
        
        features.extend([skewness, kurtosis])
        
        return np.array(features)

    def extract_edge_features(self, image):
        "Características de bordes"
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Gradientes
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features = [
            edge_density,
            np.mean(magnitude),
            np.std(magnitude),
            np.max(magnitude),
            np.sum(magnitude > np.percentile(magnitude, 90)) / magnitude.size
        ]
        
        return np.array(features)

    def extract_spatial_features(self, image):
        "Características espaciales"
        
        segments = slic(image, n_segments=self.config['slic_segments'], 
                       compactness=10, sigma=1, start_label=1)
        
        props = regionprops(segments)
        
        if not props:
            return np.zeros(15)
        
        # Extraer solo las características más importantes
        areas = [prop.area for prop in props]
        centroids = [prop.centroid for prop in props]
        eccentricities = [prop.eccentricity for prop in props]
        
        features = [
            len(props),  # Número de regiones
            np.mean(areas),
            np.std(areas),
            np.max(areas) if areas else 0,
            np.mean(eccentricities) if eccentricities else 0,
            np.std(eccentricities) if eccentricities else 0,
        ]
        
        # Distribución espacial de centroides
        if centroids:
            x_coords = [c[1] for c in centroids]
            y_coords = [c[0] for c in centroids]
            features.extend([
                np.mean(x_coords), np.std(x_coords),
                np.mean(y_coords), np.std(y_coords)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Rellenar hasta 15 características
        while len(features) < 15:
            features.append(0)
        
        return np.array(features[:15])

    def process_single_image(self, image_path):
        "Procesa una imagen con el pipeline"
        
        try:
            # Verificar cache
            if self.use_cache and image_path in self.feature_cache:
                return self.feature_cache[image_path]
            
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Redimensionar
            image = cv2.resize(image, self.image_size)
            
            # Extraer características optimizadas
            color_hist = self.extract_color_histograms(image)
            dominant_colors = self.extract_dominant_colors(image)
            texture_features = self.extract_texture(image)
            composition_features = self.extract_composition_features(image)
            statistical_features = self.extract_statistical_features(image)
            edge_features = self.extract_edge_features(image)
            spatial_features = self.extract_spatial_features(image)
            hu_moments = self.extract_hu_moments(image)
            color_moments = self.extract_color_moments(image)
            # Concatenar características
            all_features = np.concatenate([
                color_hist,
                dominant_colors,
                texture_features,
                composition_features,
                statistical_features,
                edge_features,
                spatial_features,
                hu_moments,
                color_moments 
            ])
            
            # Guardar en cache
            if self.use_cache:
                self.feature_cache[image_path] = all_features
            
            return all_features
            
        except Exception as e:
            print(f"Error procesando {image_path}: {str(e)}")
            return None

    def extract_features_parallel(self, read_csv_path='test.csv', data_dir='src/data/posters'):
        "Extrae características usando procesamiento paralelo"
        
        df = pd.read_csv(read_csv_path)
        image_paths = [os.path.join(data_dir, f"{row['movieId']}.jpg") for _, row in df.iterrows()]
        
        print(f"Procesando {len(image_paths)} imágenes con {self.n_jobs} procesos...")
        
        # Procesamiento paralelo
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            features_list = list(tqdm(
                executor.map(self.process_single_image, image_paths),
                total=len(image_paths),
                desc="Extrayendo características"
            ))
        
        # Filtrar resultados válidos
        valid_features = [f for f in features_list if f is not None]
        failed_count = len(image_paths) - len(valid_features)
        
        if not valid_features:
            raise ValueError("No se pudieron extraer características de ninguna imagen.")
        
        features = np.array(valid_features)
        print(f"Características extraídas: {features.shape}")
        
        return self._save_and_process_features(features, failed_count)

    def _save_and_process_features(self, features, failed_count):
        "Guarda y procesa las características extraídas"
        
        # normalizar características
        scaler_standard = StandardScaler()
        
        features_standard = scaler_standard.fit_transform(features)
        
        # Guardar características
        np.save('features.npy', features_standard)
        
        print(f"Características extraídas y guardadas.")
        print(f"   - Total de imágenes procesadas: {len(features)}")
        print(f"   - Imágenes fallidas: {failed_count}")
        print(f"   - Dimensiones de características: {features.shape}")
        
        return features_standard
