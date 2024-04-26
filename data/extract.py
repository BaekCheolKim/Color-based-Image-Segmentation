import tarfile
import os

# Path to your .tar.gz file
tar_path_cifar10 = 'cifar-10-python.tar.gz'
extract_path = 'data'

# Extract the tar.gz file
with tarfile.open(tar_path_cifar10, "r:gz") as tar:
    tar.extractall(path=extract_path)

with tarfile.open(tar_path_cifar100, "r:gz") as tar:
    tar.extractall(path=extract_path)
print("Extraction Done!")