import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']


def extract_features(df):
    return df[FEATURE_COLUMNS].values


def fit_scaler(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled


def transform_features(scaler, X):
    return scaler.transform(X)
