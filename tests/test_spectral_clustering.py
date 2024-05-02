import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.transform import resize
import sys
sys.path.append('D:/repos/Color-based-Image-Segmentation/src')  # Adjust path as necessary
from spectral_clustering import spectral_clustering

def load_cifar10_data(file_path, img_idx=0):
    """
    Load a specific image from CIFAR-10 dataset.
    
    Parameters:
    - file_path: str, path to the CIFAR-10 batch file
    - img_idx: int, index of the image to load
    
    Returns:
    - img: numpy array, the image data
    """
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
        img = batch['data'][img_idx]
        img = img.reshape(3, 32, 32).transpose([1, 2, 0])  # Reshape and reorder dimensions
    return img

# Load an image from the first batch
first_batch_path = 'data/cifar-10-batches-py/data_batch_1'
image = load_cifar10_data(first_batch_path, img_idx=0)  # Change img_idx to select different images

# Optionally resize the image for faster processing
image = resize(image, (64, 64), anti_aliasing=True)

# Apply the spectral clustering
segmented_image = spectral_clustering(image)

# Plot the original and segmented images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_image)
ax[1].set_title('Segmented Image')
ax[1].axis('off')

plt.show()
