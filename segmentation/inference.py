import numpy as np
import joblib


class CustomerSegmentationModel:
    def __init__(self, model_path, scaler_path, metadata_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.metadata = joblib.load(metadata_path)

    def predict(self, age, income, spending):
        arr = np.array([[age, income, spending]])
        scaled = self.scaler.transform(arr)
        cluster_id = self.model.predict(scaled)[0]
        return {
            "cluster_id": cluster_id,
            "segment_name": self.metadata['cluster_names'][cluster_id]
        }

    def get_centroids(self, real=True):
        centroids = self.model.cluster_centers_
        if real:
            centroids = self.scaler.inverse_transform(centroids)
        return centroids

    def get_all_customers(self):
        # Load your original CSV with clusters added
        df = pd.read_csv("Mall_Customers.csv")
        df['Cluster_ID'] = self.model.predict(self.scaler.transform(df[['Age','Annual Income (k$)','Spending Score (1-100)']]))
        df['Cluster_Name'] = df['Cluster_ID'].map(self.metadata['cluster_names'])
        return df

    def get_cluster_summary(self):
        df = self.get_all_customers()
        numeric_cols = df.select_dtypes(include=np.number).columns
        return df.groupby('Cluster_Name')[numeric_cols].mean().round(1)
