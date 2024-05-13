def load_cifar10images_by_label(file_path, label_name):
    import pickle
    """
    Load all images with a specified label from a CIFAR-10 batch file.
    
    Parameters:
    - file_path: str, path to the CIFAR-10 batch file
    - label_name: str, name of the label to search for
    
    Returns:
    - images: list of numpy arrays, containing all images with the specified label, shape(3, 32, 32)
    """
    label_to_class = {
        "airplane": 0, "automobile": 1, "bird": 2, "cat": 3,
        "deer": 4, "dog": 5, "frog": 6, "horse": 7,
        "ship": 8, "truck": 9
    }
    class_label = label_to_class[label_name]

    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
        images = batch['data']
        labels = batch['labels']
    
    label_images = []
    for i, label in enumerate(labels):
        if label == class_label:
            img = images[i]
            img = img.reshape(3, 32, 32).transpose([1,2,0])
            label_images.append(img)
    return label_images

def load_cifar10images_by_idx(file_path, img_idx=0):
    import pickle
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