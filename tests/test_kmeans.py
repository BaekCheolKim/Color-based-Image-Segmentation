import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from kmeans import KMeans
from load_image import load_cifar10images_by_label

size = 3
label = "horse"
images = load_cifar10images_by_label('data/cifar-10-batches-py/data_batch_1',label)

for idx, image in enumerate(images[:size]):
    pixel = (np.array(image, dtype=np.float64) / 255).reshape((-1, 3))

    # Perform K-means clustering
    Kmeans = KMeans(n_clusters=4, max_iter=200, atol=1e-4)
    segmented_img = Kmeans.fit(pixel)
    segmented_img = Kmeans.cluster_centers_[Kmeans.labels_].reshape(image.shape)

    # Create a directory to save the segmented images if it doesn't exist
    save_dir = os.path.join('kmeans_images', label)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the original and segmented images with different names based on the index number
    original_image_path = os.path.join(save_dir, f'original_image_{idx}.png')
    segmented_image_path = os.path.join(save_dir, f'segmented_image_{idx}.png')

    plt.imsave(original_image_path, image)
    plt.imsave(segmented_image_path, segmented_img)

    # Display the images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(segmented_img)
    ax[1].set_title('Segmented Image')
    ax[1].axis('off')

    plt.show()