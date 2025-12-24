from segmentation.inference import CustomerSegmentationModel

model = CustomerSegmentationModel(
    "models/kmeans_model.pkl",
    "models/scaler.pkl",
    "models/cluster_metadata.pkl"
)

result = model.predict(age=30, income=60, spending=75)
print(result)
