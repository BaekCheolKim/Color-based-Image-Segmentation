import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_rmse(original, segmented):
    return np.sqrt(mean_squared_error(original, segmented))

def compute_mae(original, segmented):
    return mean_absolute_error(original, segmented)

def compute_mape(original, segmented):
    return np.mean(np.abs((original - segmented) / original)) * 100