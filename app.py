import streamlit as st
import pandas as pd
import plotly.express as px
from segmentation.inference import predict_customer_segment, get_all_customers

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍 Customer Segmentation Dashboard")

# Sidebar for new customer input
st.sidebar.header("Predict Customer Segment")
age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.slider("Annual Income (k$)", 15, 140, 60)
spending = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

if st.sidebar.button("Predict Segment"):
    result = predict_customer_segment(age, income, spending)
    st.sidebar.success(f"Cluster ID: {result['cluster_id']}")
    st.sidebar.info(f"Segment: {result['segment_name']}")

# Load all customer data with clusters
df_clusters = get_all_customers()

# Show summary stats
st.subheader("Cluster Summary")
summary = df_clusters.groupby("Cluster_Name")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1)
st.dataframe(summary)

# 3D Visualization
st.subheader("3D Cluster Visualization")
fig = px.scatter_3d(
    df_clusters,
    x="Age",
    y="Annual Income (k$)",
    z="Spending Score (1-100)",
    color="Cluster_Name",
    hover_data=["Gender"],
    height=600
)
st.plotly_chart(fig, use_container_width=True)
