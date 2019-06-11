import numpy as np
import pandas as pd

from sicm.models.general import GeneralModel
from sicm.models import functions as funs

def scale_by_init(t, x):
    idxs = np.nonzero(t < 250e-3)[0]
    # bulk_current = np.quantile(approach_current[idxs], 0.95)
    scaler = np.quantile(x[idxs], 0.95)
    return x/scaler, scaler

def rolling_mean(x, y, window_size = 10):
    if window_size > 1:
        y_temp = pd.Series(y).rolling(window = window_size).mean()
        x_temp = x
        # Apply changes
        y = y_temp.iloc[window_size-1:].values
        x = x_temp[window_size-1:]
    return x, y

def find_bulk_val(x, y):
    """Find Value in bulk"""
    gm = GeneralModel(x, y, fun = funs._exponential_fit)
    gm.fit(guess = [1, 1, 1], verbose = False, maxfev = np.int(1e6), do_plot = False)
    x_ax = np.arange(np.min(x), np.max(x)*100, 1)
    y_hat = gm.predict(x_ax)
    # try:
    #     idx = np.argmax(np.diff(y_hat) < (np.max(y) - np.min(y)) * 1e-3)
    #     y_bulk = y_hat[idx]
    # except Exception as e:
    #     y_bulk = y_hat[-1]
    return y_hat[-1]
