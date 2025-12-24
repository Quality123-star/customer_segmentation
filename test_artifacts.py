import joblib

model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")
cluster_names = joblib.load("models/cluster_metadata.pkl")

print("Model:", type(model))
print("Scaler:", type(scaler))
print("Cluster names:", cluster_names)
