import tarfile
import kmeans
import numpy as np

tar_path_cifar10 = 'cifar-10-python.tar.gz'
extract_path = 'data'

with tarfile.open(tar_path_cifar10, "r:gz") as tar:
    tar.extractall(path=extract_path)

image = None
image = np.array(image, dtype=np.float64) / 255
pixels = image.reshape((-1, 3))
cluster = 5

# Perform K-means clustering
Kmeans = kmeans.KMeans(n_clusters=cluster)
Kmeans.fit(pixels)
labels = Kmeans.labels_
centroids = Kmeans.cluster_centers_

# Assign each pixel the color of its nearest centroid
segmented_img = centroids[labels].reshape(image.shape)