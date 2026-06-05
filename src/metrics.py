import numpy as np

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(
        np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)
    )