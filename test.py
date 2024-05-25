import numpy as np

# Define the updated Laplacian matrix
L = np.array([[np.exp(-2), 0, 0, -np.exp(-2), 0],
              [0, np.exp(-2), -np.exp(-2), 0, 0],
              [0, -np.exp(-2), 2*np.exp(-2), 0, -np.exp(-2)],
              [-np.exp(-2), 0, 0, np.exp(-2), 0],
              [0, 0, -np.exp(-2), 0, np.exp(-2)]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(L)
print("eigenvalues size : ", eigenvalues.shape)
print("eigenvalues : ", eigenvalues)
print("eigenvectors size : ", eigenvectors.shape)
print("eigenvectors : ", eigenvectors)

# Second smallest eigenvalue and its eigenvector
secondSmallestEigenvalue = np.sort(eigenvalues)[1]
secondSmallestEigenvector = eigenvectors[:, np.argsort(eigenvalues)[1]]

print("secondSmallestEigenvalue : ", secondSmallestEigenvalue)
print("secondSmallestEigenvector : ", secondSmallestEigenvector)
