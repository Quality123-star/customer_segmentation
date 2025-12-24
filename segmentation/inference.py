import os
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")
CLUSTER_METADATA_PATH = os.path.join(MODELS_DIR, "cluster_metadata.pkl")
CSV_PATH = os.path.join(DATA_DIR, "Mall_Customers.csv")

scaler = joblib.load(SCALER_PATH)
kmeans_model = joblib.load(MODEL_PATH)
cluster_names = joblib.load(CLUSTER_METADATA_PATH)


def predict_customer_segment(age, income, spending):
    X = np.array([[age, income, spending]])
    X_scaled = scaler.transform(X)
    cluster_id = kmeans_model.predict(X_scaled)[0]
    return {
        "cluster_id": int(cluster_id),
        "segment_name": cluster_names[cluster_id]
    }


def get_all_customers():
    df = pd.read_csv(CSV_PATH)
    X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    X_scaled = scaler.transform(X)
    df["Cluster_ID"] = kmeans_model.predict(X_scaled)
    df["Cluster_Name"] = df["Cluster_ID"].map(cluster_names)
    return df
