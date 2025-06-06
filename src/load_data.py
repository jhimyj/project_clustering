import numpy as np
import pandas as pd

def load_data(features_path, metadata_path, labels_path = None):
    features = np.load(features_path)
    metadata = pd.read_csv(metadata_path)
    if labels_path is not None:
        labels = np.load(labels_path)
        metadata["cluster"] = labels
    return features, metadata