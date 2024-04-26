import unittest
import numpy as np
from src.k_means_clustering import load_cifar10_batch, preprocess_images, k_means_segmentation

class TestKMeansClustering(unittest.TestCase):
    def test_load_cifar10_batch(self):
        data, labels = load_cifar10_batch('data/cifar-10-batches-py/data_batch_1')
        self.assertEqual(data.shape, (10000, 3072))  # Assuming the data batch has 10,000 images each of 32*32*3
        self.assertEqual(len(labels), 10000)

    def test_preprocess_images(self):
        data = np.random.randint(0, 255, (10, 3072))
        processed_data = preprocess_images(data)
        self.assertEqual(processed_data.shape, (10, 1024, 3))  # 32*32*3 reshaped to 32*32, 3 channels
        self.assertTrue((processed_data <= 1).all() and (processed_data >= 0).all())

    def test_k_means_segmentation(self):
        data = np.random.rand(10, 32*32, 3)  # Randomly generated data as if it was preprocessed
        segmented_data = k_means_segmentation(data)
        self.assertEqual(segmented_data.shape, data.shape)  # Output shape should match input shape

    def test_k_means_results_consistency(self):
        data = np.random.rand(10, 32*32, 3)  # Consistent input data
        segmented_data1 = k_means_segmentation(data)
        segmented_data2 = k_means_segmentation(data)
        np.testing.assert_array_almost_equal(segmented_data1, segmented_data2, decimal=5)  # Results should be consistent for the same input

if __name__ == '__main__':
    unittest.main()
