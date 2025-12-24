from sklearn.cluster import KMeans


def train_kmeans(X_scaled, n_clusters=5, random_state=42):
    model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=random_state,
        n_init=10
    )
    labels = model.fit_predict(X_scaled)
    return model, labels
