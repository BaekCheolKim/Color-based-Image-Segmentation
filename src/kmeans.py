import numpy as np


class KMeans:
    """
    Display segmented image: segmented_img = centroids[labels].reshape(img.shape)
    """
    def __init__(self, n_clusters, max_iter=100):
        """Initialise k means algorithm with required number of clusters and maximum interations"""
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        """Perform K means algorithm for given n_cluster on float 64 image normalised and reshaped into single vector"""
        # Randomly initialize centroids
        centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            # Assign each data point to the nearest centroid
            labels = np.argmin(((X[:, None] - centroids)**2).sum(axis=2), axis=1)
            # Update centroids by taking the mean of all data points assigned to each centroid
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            # Check for convergence
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        self.labels_ = labels
        self.cluster_centers_ = centroids