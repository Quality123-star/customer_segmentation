from segmentation.inference import predict_customer_segment, get_all_customers
import pandas as pd

# Sample batch of customers
customers = pd.DataFrame([
    {"Age": 22, "Income": 30, "Spending": 80},
    {"Age": 45, "Income": 100, "Spending": 20},
    {"Age": 35, "Income": 70, "Spending": 60}
])

# Predict each customer
for _, row in customers.iterrows():
    result = predict_customer_segment(row.Age, row.Income, row.Spending)
    print(result)
