import numpy as np
import joblib


class CustomerSegmentationModel:
    def __init__(self, model_path, scaler_path, metadata_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.cluster_names = joblib.load(metadata_path)

    def predict(self, age, income, spending):
        X = np.array([[age, income, spending]])
        X_scaled = self.scaler.transform(X)
        cluster_id = self.model.predict(X_scaled)[0]
        return {
            "cluster_id": int(cluster_id),
            "segment_name": self.cluster_names[cluster_id]
        }
