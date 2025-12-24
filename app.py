import streamlit as st
import pandas as pd
import plotly.express as px
from segmentation.inference import predict_customer_segment, get_all_customers

from segmentation.inference import (
    predict_customer_segment,
    get_all_customers,
    get_cluster_centroids
)


st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍 Customer Segmentation Dashboard")

tab1, tab2, tab3 = st.tabs(
    ["📊 Dashboard", "🔍 Model Explanation", "📁 Data"]
)


with tab1:
    st.sidebar.header("Predict Customer Segment")

    age = st.sidebar.slider("Age", 18, 70, 30)
    income = st.sidebar.slider("Annual Income (k$)", 15, 140, 60)
    spending = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

    if st.sidebar.button("Predict Segment"):
        result = predict_customer_segment(age, income, spending)
        st.sidebar.success(f"Cluster ID: {result['cluster_id']}")
        st.sidebar.info(f"Segment: {result['segment_name']}")

    df_clusters = get_all_customers()

    st.subheader("Cluster Summary")
    summary = df_clusters.groupby("Cluster_Name")[
        ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    ].mean().round(1)

    st.dataframe(summary)

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


with tab2:
    st.header("🔍 How the Model Works")

    st.markdown("""
    ### 🎯 Problem Being Solved
    This model groups mall customers into distinct segments based on
    similarities in their **age**, **income**, and **spending behavior**.
    
    The goal is to help businesses:
    - Understand different customer types
    - Personalize marketing strategies
    - Improve customer targeting
    """)

    st.markdown("""
    ### 🤖 Algorithm: K-Means Clustering
    K-Means is an **unsupervised learning algorithm**, meaning:
    - There are **no predefined labels**
    - The model discovers patterns automatically
    
    It works by:
    1. Placing *K* cluster centers in the data space
    2. Assigning each customer to the nearest center
    3. Repeating until cluster positions stabilize
    """)

    st.markdown("""
    ### 🧮 Features Used
    The model uses the following customer attributes:
    
    - **Age** – captures life-stage differences
    - **Annual Income (k$)** – reflects purchasing power
    - **Spending Score (1–100)** – indicates spending behavior
    
    All features are **standardized** before clustering to ensure fairness.
    """)

    st.markdown("### 📌 Cluster Centroids (Typical Customer per Cluster)")
    centroids = get_cluster_centroids().round(1)
    centroids["Cluster"] = centroids.index
    st.dataframe(centroids)

    st.markdown("""
    ### 📏 How a Prediction Is Made
    When a new customer is entered:
    
    - Their data is **scaled**
    - Distance to each cluster centroid is calculated
    - The **nearest cluster** is assigned
    
    The segment name is a **human-readable summary** of that cluster’s profile.
    """)

    st.info(
        "⚠️ Clusters describe **groups**, not individuals. "
        "Customers near cluster boundaries may share characteristics with multiple segments."
    )

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif",
    caption="K-Means clustering process (illustrative)"
)
