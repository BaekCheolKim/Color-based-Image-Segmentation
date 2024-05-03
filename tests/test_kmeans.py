import pickle
import kmeans
import numpy as np
import matplotlib.pyplot as plt

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
        img = img.reshape(3, 32, 32).transpose([1, 2, 0])
    return img

image = load_cifar10_data('data/cifar-10-batches-py/data_batch_1',0)
iamge = (np.array(image, dtype=np.float64) / 255).reshape((-1,3))

# Perform K-means clustering
Kmeans = kmeans.KMeans(n_clusters=4, max_iter=200, atol=1e-4)
segmented_img = Kmeans.fit(image)

# Assign each pixel the color of its nearest centroid
segmented_img = Kmeans.cluster_centers_[Kmeans.labels_].reshape(image.shape)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_img)
ax[1].set_title('Segmented Image')
ax[1].axis('off')

plt.show()