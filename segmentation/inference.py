import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load saved artifacts
BASE_DIR = os.path.dirname(__file__)  # folder of this file
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")
CLUSTER_METADATA_PATH = os.path.join(MODELS_DIR, "cluster_metadata.pkl")
CSV_PATH = os.path.join(BASE_DIR, "..", "Mall_Customers.csv")  # CSV relative to project root

# Load scaler, model, cluster names
scaler = joblib.load(SCALER_PATH)
kmeans_model = joblib.load(MODEL_PATH)
cluster_names = joblib.load(CLUSTER_METADATA_PATH)

# Function to predict a single customer
def predict_customer_segment(age, income, spending):
    new_customer = np.array([[age, income, spending]])
    new_customer_scaled = scaler.transform(new_customer)
    cluster_id = kmeans_model.predict(new_customer_scaled)[0]
    segment_name = cluster_names[cluster_id]
    return {"cluster_id": cluster_id, "segment_name": segment_name}

# Function to get all customers with cluster assignment
def get_all_customers():
    df = pd.read_csv(CSV_PATH)
    X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values
    X_scaled = scaler.transform(X)
    df["Cluster_ID"] = kmeans_model.predict(X_scaled)
    df["Cluster_Name"] = df["Cluster_ID"].map(cluster_names)
    return df
