import numpy as np
from keras.datasets import cifar10
from skimage import feature, color
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.sparse.linalg import eigsh

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
desired_labels = ['airplane', 'deer', 'horse']
desired_indices = [labels.index(label) for label in desired_labels]

# Filter training and test sets
train_mask = np.isin(y_train, desired_indices).flatten()
X_train_filtered = X_train[train_mask]
y_train_filtered = y_train[train_mask]

test_mask = np.isin(y_test, desired_indices).flatten()
X_test_filtered = X_test[test_mask]
y_test_filtered = y_test[test_mask]

def extract_multiscale_features(image, scales=[1, 0.5, 0.25]):
    h, w, c = image.shape
    features = []
    for scale in scales:
        scaled_img = cv2.resize(image, (int(w * scale), int(h * scale)))
        gray = color.rgb2gray(scaled_img)
        color_features = scaled_img.reshape(-1, 3)
        lbp = feature.local_binary_pattern(gray, P=8, R=1.0)
        lbp_features = lbp.reshape(-1, 1)
        combined_features = np.hstack((color_features, lbp_features))
        if scale != 1:
            combined_features = cv2.resize(combined_features, (w, h)).reshape(h * w, -1)
        features.append(combined_features)
    multiscale_features = np.hstack(features)
    return multiscale_features

def compute_similarity_graph(features, sigma=1.0):
    W = rbf_kernel(features, gamma=1.0 / (2 * sigma ** 2))
    return W

def compute_laplacian(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    L = (L + L.T) / 2  # Ensuring symmetry
    return L

def spectral_embedding(L, k=4, max_iter=5000, tol=1e-5):
    eigenvalues, eigenvectors = eigsh(L, k=k, which='SM', maxiter=max_iter, tol=tol)
    return eigenvectors

def cluster_embedding(embedding, k=4):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(embedding)
    return labels

def compare_clustering_algorithms(embedding, k=4):
    kmeans_labels = KMeans(n_clusters=k).fit_predict(embedding)
    agglomerative_labels = AgglomerativeClustering(n_clusters=k).fit_predict(embedding)
    dbscan_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(embedding)
    return kmeans_labels, agglomerative_labels, dbscan_labels

def plot_segmented_images(original, segmented_images, titles):
    fig, axes = plt.subplots(1, len(segmented_images) + 1, figsize=(20, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, (segmented, title) in enumerate(zip(segmented_images, titles)):
        axes[i + 1].imshow(segmented, cmap='viridis')
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')

    plt.show()

# Example usage
sample_image = X_train_filtered[0]
features = extract_multiscale_features(sample_image)
W = compute_similarity_graph(features)
L = compute_laplacian(W)
embedding = spectral_embedding(L)
labels = cluster_embedding(embedding)
segmented_image = labels.reshape(sample_image.shape[:2])

kmeans_labels, agglomerative_labels, dbscan_labels = compare_clustering_algorithms(embedding)

kmeans_segmented_image = kmeans_labels.reshape(sample_image.shape[:2])
agglomerative_segmented_image = agglomerative_labels.reshape(sample_image.shape[:2])
dbscan_segmented_image = dbscan_labels.reshape(sample_image.shape[:2])

plot_segmented_images(
    sample_image,
    [segmented_image, kmeans_segmented_image, agglomerative_segmented_image, dbscan_segmented_image],
    ["Spectral Clustering", "K-means", "Agglomerative Clustering", "DBSCAN"]
)
