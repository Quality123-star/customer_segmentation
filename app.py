import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from segmentation.inference import CustomerSegmentationModel

# Load artifacts
model = CustomerSegmentationModel(
    "models/kmeans_model.pkl",
    "models/scaler.pkl",
    "models/cluster_metadata.pkl"
)

st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("🛍️ Customer Segmentation Dashboard")

# --- Sidebar: Input customer details ---
st.sidebar.header("Predict Customer Segment")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.slider("Annual Income (k$)", 10, 150, 60)
spending = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

if st.sidebar.button("Predict Segment"):
    result = model.predict(age, income, spending)
    st.success(f"Predicted Segment: **{result['segment_name']}** (Cluster ID: {result['cluster_id']})")

# --- Load customer dataset ---
st.sidebar.header("Load CSV for batch predictions")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    batch_results = []
    for _, row in df.iterrows():
        pred = model.predict(row['Age'], row['Annual Income (k$)'], row['Spending Score (1-100)'])
        batch_results.append(pred)
    batch_df = df.copy()
    batch_df['Cluster_ID'] = [r['cluster_id'] for r in batch_results]
    batch_df['Segment_Name'] = [r['segment_name'] for r in batch_results]
    st.write("Batch Predictions", batch_df)

# --- 3D Visualization ---
st.header("3D Cluster Visualization")
centroids = model.get_centroids(real=True)  # real, unscaled values
df_clusters = model.get_all_customers()    # Returns df with Cluster_ID column

fig = px.scatter_3d(
    df_clusters,
    x="Age",
    y="Annual Income (k$)",
    z="Spending Score (1-100)",
    color="Cluster_Name",
    size_max=8,
    opacity=0.8,
    hover_data=["Age", "Annual Income (k$)", "Spending Score (1-100)"]
)

fig.add_scatter3d(
    x=centroids[:, 0],
    y=centroids[:, 1],
    z=centroids[:, 2],
    mode="markers",
    marker=dict(size=10, color='black', symbol='x'),
    name="Centroids"
)

st.plotly_chart(fig, use_container_width=True)

# --- Cluster Summary ---
st.header("Cluster Summary Statistics")
cluster_summary = model.get_cluster_summary()
st.dataframe(cluster_summary)
