import unittest
import numpy as np
from src.spectral_clustering import load_cifar10_batch, preprocess_images, spectral_segmentation

class TestSpectralClustering(unittest.TestCase):
    def test_load_cifar10_batch(self):
        data, labels = load_cifar10_batch('data/cifar-10-batches-py/data_batch_1')
        self.assertEqual(data.shape, (10000, 3072))  # Confirm data shape for CIFAR-10 batch
        self.assertEqual(len(labels), 10000)  # Confirm label count

    def test_preprocess_images(self):
        data = np.random.randint(0, 255, (10, 3072))
        processed_data = preprocess_images(data)
        self.assertEqual(processed_data.shape, (10, 1024, 3))  # Confirm reshaped and normalized data
        self.assertTrue((processed_data <= 1).all() and (processed_data >= 0).all())  # Data should be normalized

    def test_spectral_segmentation(self):
        data = np.random.rand(10, 32*32, 3)  # Random data simulating preprocessed images
        segmented_data = spectral_segmentation(data)
        self.assertEqual(segmented_data.shape, data.shape)  # Segmentation output should match input

    def test_spectral_results_consistency(self):
        data = np.random.rand(10, 32*32, 3)  # Consistent input for testing
        segmented_data1 = spectral_segmentation(data)
        segmented_data2 = spectral_segmentation(data)
        np.testing.assert_array_almost_equal(segmented_data1, segmented_data2, decimal=5)  # Check for deterministic results

if __name__ == '__main__':
    unittest.main()
