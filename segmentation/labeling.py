import numpy as np
import pandas as pd


def compute_cluster_summary(df, cluster_col):
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.groupby(df[cluster_col]).mean()



def generate_thresholds(df):
    return {
        'age_mean': df['Age'].mean(),
        'income_mean': df['Annual Income (k$)'].mean(),
        'spending_mean': df['Spending Score (1-100)'].mean()
    }


def label_age(age, mean):
    if age < mean - 5:
        return "Young"
    elif age > mean + 5:
        return "Senior"
    return "Middle-aged"


def label_income(value, mean):
    if value < mean * 0.85:
        return "Low Income"
    elif value > mean * 1.15:
        return "High Income"
    return "Mid Income"


def label_spending(value, mean):
    if value < mean * 0.85:
        return "Low Spending"
    elif value > mean * 1.15:
        return "High Spending"
    return "Mid Spending"


def generate_cluster_names(cluster_summary, thresholds):
    names = {}

    for cluster_id, row in cluster_summary.iterrows():
        name = (
            f"{label_age(row['Age'], thresholds['age_mean'])} · "
            f"{label_income(row['Annual Income (k$)'], thresholds['income_mean'])} · "
            f"{label_spending(row['Spending Score (1-100)'], thresholds['spending_mean'])}"
        )
        names[cluster_id] = name

    return names
