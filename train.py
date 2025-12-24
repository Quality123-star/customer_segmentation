import pandas as pd
import os
import joblib


from segmentation.preprocessing import extract_features, fit_scaler
from segmentation.training import train_kmeans
from segmentation.labeling import compute_cluster_summary, generate_thresholds, generate_cluster_names


# Load data
df = pd.read_csv("data/Mall_Customers.csv")

# Preprocess
X = extract_features(df)
scaler, X_scaled = fit_scaler(X)

# Train
model, labels = train_kmeans(X_scaled)

df['Cluster'] = labels

# Generate names
summary = compute_cluster_summary(df, 'Cluster')
thresholds = generate_thresholds(df)
cluster_names = generate_cluster_names(summary, thresholds)

os.makedirs("models", exist_ok=True)


# Save artifacts
joblib.dump(model, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(cluster_names, "models/cluster_metadata.pkl")

print("Training complete. Model saved.")
