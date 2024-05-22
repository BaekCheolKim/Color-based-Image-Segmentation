import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from rsme_mae_mape import compute_mae, compute_mape, compute_rmse
from load_image import load_cifar10images_by_label
from kmeans import KMeans
from spectral_clustering import spectral_clustering

label = "deer"
images = load_cifar10images_by_label('data/cifar-10-batches-py/data_batch_1', label)
image = images[0]

# Normalize the image and reshape for k-means
X = (np.array(image, dtype=np.float64) / 255).reshape((-1, 3))

# Perform k-means clustering
kmeans = KMeans(n_clusters=4, max_iter=100, atol=1e-4)
kmeans.fit(X)

# Create the segmented image for KMeans
segmented_X_kmeans = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)

# Perform spectral clustering
segmented_X_spectral = spectral_clustering(image, n_clusters=4)

print(f"Kmeans flat: {segmented_X_kmeans.shape}")
print(f"Spectral flat: {segmented_X_spectral.shape}")

# Flatten the images to 2D arrays for metric computation
original_flat = image.reshape(-1, 3) / 255.0
segmented_flat_kmeans = segmented_X_kmeans.reshape(-1, 3) / 255
# segmented_flat_spectral = segmented_X_spectral.reshape(-1, 3) / 255.0

# Compute RMSE, MAE, and MAPE for KMeans
rmse_kmeans = compute_rmse(original_flat, segmented_flat_kmeans)
mae_kmeans = compute_mae(original_flat, segmented_flat_kmeans)
mape_kmeans = compute_mape(original_flat, segmented_flat_kmeans)

# Compute RMSE, MAE, and MAPE for Spectral Clustering
# PROBLEM - (32 32 4) Additonal channel?
# rmse_spectral = compute_rmse(original_flat, segmented_flat_spectral)
# mae_spectral = compute_mae(original_flat, segmented_flat_spectral)
# mape_spectral = compute_mape(original_flat, segmented_flat_spectral)

# Display results for KMeans
print(f"KMeans - RMSE: {rmse_kmeans}")
print(f"KMeans - MAE: {mae_kmeans}")
print(f"KMeans - MAPE: {mape_kmeans:.2f}%")

# Display results for Spectral Clustering
# print(f"Spectral Clustering - RMSE: {rmse_spectral}")
# print(f"Spectral Clustering - MAE: {mae_spectral}")
# print(f"Spectral Clustering - MAPE: {mape_spectral:.2f}%")

# Plot original and segmented images
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow((segmented_X_kmeans * 255).astype(np.uint8))
ax[1].set_title('KMeans Segmented Image')
ax[1].axis('off')
ax[2].imshow(segmented_X_spectral)
ax[2].set_title('Spectral Clustering Segmented Image')
ax[2].axis('off')
plt.show()