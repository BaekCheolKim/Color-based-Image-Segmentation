import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import pickle
import os

def load_cifar10_batches(directory):
    # Initialize arrays to hold data and labels
    full_data = []
    full_labels = []
    # Loop through all batch files
    for i in range(1, 6):  # There are 5 data batches in CIFAR-10
        file = os.path.join(directory, f'data_batch_{i}')
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        full_data.append(dict[b'data'])
        full_labels.append(dict[b'labels'])
    # Convert lists to numpy arrays and concatenate
    full_data = np.concatenate(full_data)
    full_labels = np.concatenate(full_labels)
    return full_data, full_labels

def preprocess_images(data):
    # Normalize and reshape the data
    data = data / 255.0
    return data.reshape(data.shape[0], 32 * 32, 3)

def k_means_segmentation(data, n_clusters=4):
    # Flatten the data for clustering
    data_flat = data.reshape(data.shape[0] * data.shape[1], 3)
    # Sample a subset of data to fit K-means
    data_sample = shuffle(data_flat, random_state=0)[:1000]
    # Create and fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_sample)
    # Predict clusters for all data
    labels = kmeans.predict(data_flat)
    segmented_data = kmeans.cluster_centers_[labels].reshape(data.shape)
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
    # Load all data batches from CIFAR-10
    data, labels = load_cifar10_batches('data/cifar-10-batches-py/')
    # Select a subset of images for demonstration, e.g., first 100 images
    selected_images = preprocess_images(data[:100])

    for i in range(5):  # Display first 5 images and their segmentations
        original_img = selected_images[i].reshape(32, 32, 3)
        segmented_img = k_means_segmentation(selected_images[i:i+1])
        display_images(original_img, segmented_img)
