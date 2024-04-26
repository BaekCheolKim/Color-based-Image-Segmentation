import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

extract_path = 'data'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

data_path = os.path.join(extract_path, 'cifar-10-batches-py', 'data_batch_1')
batch_1 = unpickle(data_path)

images = batch_1['data']
labels = batch_1['labels']

images = images.reshape((len(images), 3, 32, 32)).transpose(0, 2, 3, 1) / 255.0  # Normalize the images

def k_means_custom(image, K=4, iterations=10):
    def initialize_centroids():
        pixels = image.reshape(-1, 3)
        indices = np.random.choice(pixels.shape[0], K, replace=False)
        return pixels[indices]

    def assign_clusters(centroids):
        pixels = image.reshape(-1, 3)
        distances = np.sqrt(((pixels - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def update_centroids(labels):
        pixels = image.reshape(-1, 3)
        return np.array([pixels[labels == k].mean(axis=0) for k in range(K)])

    centroids = initialize_centroids()
    for _ in range(iterations):
        labels = assign_clusters(centroids)
        new_centroids = update_centroids(labels)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels.reshape(image.shape[:-1]), centroids

fig, axes = plt.subplots(1, 5, figsize=(15, 3))  # Display 5 images
for i, ax in enumerate(axes):
    labels, centroids = k_means_custom(images[i])
    segmented_image = centroids[labels]
    ax.imshow(segmented_image.astype('uint8'))
    ax.set_title(f'Image {i+1}')
plt.show()
