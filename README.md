# Color-based Image Segmentation Project

## Project Idea

The goal of this project is to implement and compare two color-based image segmentation techniques on the CIFAR-10 dataset. Specifically, we will develop:

1. **K-means Clustering for K-class Image Segmentation**:
   - Implement a custom version of the K-means clustering algorithm.
   - Segment images into K=4 color-based classes.

2. **Spectral Clustering for K-class Image Segmentation**:
   - Develop a custom version of the Spectral clustering algorithm (also known as the Normalized-cut algorithm).
   - Perform segmentation into K=4 color-based classes using this method.

## Testing

The algorithms will be tested on images of horses, deer, and airplanes from the CIFAR-10 dataset. The segmentation results obtained with both K-means and Spectral clustering will be compared to evaluate their performance.

## Useful Materials

For reference and a better understanding of the algorithms, the following materials may be consulted:

- J. Shi and J. Malik, "Normalized Cuts and Image Segmentation," Proc. IEEE Conf. Computer Vision and Pattern Recognition, pp. 731-737, 1997.
- J. Shi and J. Malik, "Normalized Cuts and Image Segmentation," IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 22, No. 8, August 2000.

## Dataset

The CIFAR-10 dataset used for this project can be found at the following link:
https://www.cs.toronto.edu/~kriz/cifar.html

## How to Run
# K-means Algorithm
(Instructions on how to run the implementations would be provided here.)
Intilise K-means algorithm using the class name and specifing number of clusters and maximum interations.
Ensure image is of float 64, normalised into range [0 1] and reshaped into single vector using reshape(-1,3).
Use cluster_centers_ and labels_ to produce segmented image and reshape before viewing.

kmeans = KMeans(n_clusters=cluster)
kmeans.fit(pixels)
segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)

## Results

(Section to include the results and comparisons of the segmentation techniques.)

## Conclusion

(A brief conclusion based on the findings and comparisons of the different segmentation methods.)
