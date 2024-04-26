import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import pickle
import os

def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']

def preprocess_images(data):
    # Normalize and reshape the data
    data = data / 255.0
    data = data.reshape(data.shape[0], 32 * 32, 3)
    return data

def spectral_segmentation(data, n_clusters=4):
    # Reshape the data for clustering
    data_flat = data.reshape(data.shape[0] * data.shape[1], 3)
    # Perform Spectral Clustering
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)
    labels = clustering.fit_predict(data_flat)
    segmented_data = data_flat[labels].reshape(data.shape)
    return segmented_data

def display_images(original, segmented):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(original)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(segmented)
    ax[1].set_title('Segmented Image')
    ax[1].axis('off')
    plt.show()

if __name__ == '__main__':
    # Load a single batch from CIFAR-10
    data, labels = load_cifar10_batch('data/cifar-10-batches-py/data_batch_1')
    # Select a subset of images for demonstration, e.g., first 100 images
    selected_images = preprocess_images(data[:100])

    for i in range(5):  # Display first 5 images and their segmentations
        original_img = selected_images[i].reshape(32, 32, 3)
        segmented_img = spectral_segmentation(selected_images[i:i+1])
        display_images(original_img, segmented_img)
