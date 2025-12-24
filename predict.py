from segmentation.inference import predict_customer_segment, get_all_customers

result = predict_customer_segment(age=30, income=70, spending=50)
print(result)

