import numpy as np

def MSE(y_gt, y_pred):
    return np.mean(np.power(y_gt - y_pred, 2))

def MSE_prim(y_gt, y_pred):
    return 2* (y_pred - y_gt) / np.size(y_gt)    