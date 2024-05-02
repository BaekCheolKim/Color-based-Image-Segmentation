import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist


"""
Perform spectral clustering with feature enhancement and spectral rotation.

Parameters:
- image: numpy array, shape (height, width, channels), input image
- n_clusters: int, number of desired clusters
- sigma_color: float, scaling factor for color distances
- sigma_space: float, scaling factor for spatial distances

Returns:
- segmented_image: numpy array, segmented image
"""

def spectral_clustering(image, n_clusters=4, sigma_color=10.0, sigma_space=0.5):

    # Normalizing and reshaping the image
    img_flat = (image / 255.0).reshape(-1, 3)
    height, width, channels = image.shape
    x, y = np.indices((height, width))
    
    # Feature enhancement: combining color and spatial features
    features = np.concatenate((img_flat, x.reshape(-1, 1) / sigma_space, y.reshape(-1, 1) / sigma_space), axis=1)
    
    # Computing the affinity matrix
    dists = cdist(features, features, 'euclidean')
    affinity_matrix = np.exp(-dists / sigma_color**2)
    
    # Eigen decomposition for dimensionality reduction
    eigenvalues, eigenvectors = eigsh(affinity_matrix, k=n_clusters + 1, which='LM', sigma=0.0)
    
    # Optional: Spectral rotation (modify this according to your needs)
    # For example, here a simple normalization is applied
    U = eigenvectors[:, 1:]  # skip the first eigenvector for better clustering
    norm_U = U / np.linalg.norm(U, axis=1, keepdims=True)
    
    # Clustering using KMeans on the transformed features
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(norm_U)
    labels = kmeans.labels_
    
    # Assign each pixel the color of its nearest centroid
    segmented_image = kmeans.cluster_centers_[labels].reshape(height, width, -1) * 255
    return segmented_image.astype(np.uint8)