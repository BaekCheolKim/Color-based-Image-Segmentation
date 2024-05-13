import numpy as np


class KMeans:
    """
    Display segmented image: segmented_img = centroids[labels].reshape(img.shape)
    """
    def __init__(self, n_clusters, max_iter=100, atol=1e-4):
        """
        Initialize KMeans clustering algorithm.
        
        Parameters:
        - n_clusters: int, number of clusters (k)
        - max_iter: int, maximum number of iterations
        - atol: float, tolerance to declare convergence
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.atol = atol

    def fit(self, X):
        """
        Fit KMeans to the data.
        
        Parameters:
        - X: array-like, shape (n_samples, n_features), input data

        Returns:
        - segmented_image: array-like, shape (n_samples, n_features), output data
        """
        centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            labels = np.argmin(((X[:, None] - centroids)**2).sum(axis=2), axis=1)
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.allclose(new_centroids, centroids, atol = self.atol):
                break
            centroids = new_centroids
        self.labels_ = labels
        self.cluster_centers_ = centroids

        return self.cluster_centers_[self.labels_].reshape(X.shape)