from segmentation.inference import CustomerSegmentationModel
import pandas as pd

model = CustomerSegmentationModel(
    "models/kmeans_model.pkl",
    "models/scaler.pkl",
    "models/cluster_metadata.pkl"
)

customers = pd.DataFrame([
    {"Age": 22, "Income": 30, "Spending": 80},
    {"Age": 45, "Income": 100, "Spending": 20},
    {"Age": 35, "Income": 70, "Spending": 60}
])

for _, row in customers.iterrows():
    print(model.predict(row.Age, row.Income, row.Spending))
