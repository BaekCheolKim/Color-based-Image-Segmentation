import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics.pairwise import euclidean_distances as cdist
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import src.spectral_clustering

def load_cifar10_categories(base_dir, batches, categories):
    """
    Load specific categories from specified CIFAR-10 batch files.
    
    Args:
    - base_dir (str): Directory containing CIFAR-10 batch files.
    - batches (list): List of batch filenames to load.
    - categories (list): List of category indices to load.
    
    Returns:
    - np.array: Filtered images.
    - np.array: Corresponding labels.
    """
    x_filtered = []
    y_filtered = []
    
    for batch in batches:
        batch_path = os.path.join(base_dir, batch)
        with open(batch_path, 'rb') as file:
            data_dict = pickle.load(file, encoding='bytes')
            images = data_dict[b'data']
            labels = np.array(data_dict[b'labels'])
            
            # Reshape the images
            images = images.reshape(len(images), 3, 32, 32).transpose(0, 2, 3, 1)
            
            # Filter images and labels based on categories
            mask = np.isin(labels, categories)
            x_filtered.append(images[mask])
            y_filtered.append(labels[mask])
    
    x_filtered = np.concatenate(x_filtered, axis=0)
    y_filtered = np.concatenate(y_filtered, axis=0)
    
    return x_filtered, y_filtered

def plot_sample_images(data, labels, categories, num_samples=5):
    """
    Plots a number of sample images for each specified category.
    
    Args:
    - data (np.array): The image data.
    - labels (np.array): The labels corresponding to the image data.
    - categories (list): Categories to display.
    - num_samples (int): Number of samples to display per category.
    """
    fig, axs = plt.subplots(len(categories), num_samples, figsize=(num_samples * 2, len(categories) * 2))
    
    for idx, category in enumerate(categories):
        indices = np.where(labels == category)[0]
        indices = np.random.choice(indices, num_samples, replace=False)  # Select random samples
        
        for j, image_index in enumerate(indices):
            ax = axs[idx, j] if len(categories) > 1 else axs[j]
            ax.imshow(data[image_index])
            ax.axis('off')
            if j == 0:
                ax.set_title(f"Category {category}")
    plt.show()

# Example usage:
base_dir = 'data/cifar-10-batches-py'
batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']  # specify the batches you want to load
categories = [0, 4, 7]  # 0: airplane, 4: deer, 7: horse

x_filtered, y_filtered = load_cifar10_categories(base_dir, batches, categories)

print(x_filtered.shape, y_filtered.shape)
plot_sample_images(x_filtered, y_filtered, categories)

def plot_comparison(original_images, clustered_images, num_samples=5):
    """
    Plots a comparison of original and clustered images.
    """
    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    for i in range(num_samples):
        # Display original image
        axs[0, i].imshow(original_images[i])
        axs[0, i].axis('off')
        axs[0, i].set_title("Original")

        # Display clustered image
        axs[1, i].imshow(clustered_images[i])
        axs[1, i].axis('off')
        axs[1, i].set_title("Clustered")
    plt.show()

def apply_spectral_clustering(images, n_clusters=4):
    """
    Apply spectral clustering to a set of images.
    """
    clustered_images = []
    for image in images:
        clustered_image = spectral_clustering(image, n_clusters=n_clusters)
        clustered_images.append(clustered_image)
    return clustered_images

# Example usage:
base_dir = 'data/cifar-10-batches-py'
batches = ['data_batch_1']  # Use one batch for simplicity
categories = [0, 4, 7]  # Airplane, deer, horse

x_filtered, y_filtered = load_cifar10_categories(base_dir, batches, categories)

# Applying spectral clustering to a subset of images
n_clusters = 4
sampled_images = x_filtered[:5]  # Select first 5 images for demonstration
clustered_images = apply_spectral_clustering(sampled_images, n_clusters)

# Plotting the comparison
plot_comparison(sampled_images, clustered_images)
