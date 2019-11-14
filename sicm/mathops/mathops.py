import numpy as np
import pandas as pd

from sicm.models.general import GeneralModel
from sicm.models import functions as funs
from scipy.stats import trim_mean
from sicm.plots import plots

def scale_by_init(t, x, how = "quantile", q = 0.95, t_len = 250e-3):
    """Scale values by initial measurements acquired in t_len interval"""
    idxs = np.nonzero(t <= t_len)[0]
    # bulk_current = np.quantile(approach_current[idxs], 0.95)
    if how.lower().startswith("q"):
        scaler = np.quantile(x[idxs], q)
    elif how.lower().startswith("t"):
        scaler = trim_mean(x[idxs], q)
    return x/scaler, scaler

def rolling_mean(x, y, window_size = 10):
    if window_size > 1:
        y_temp = pd.Series(y).rolling(window = window_size).mean()
        # Apply changes
        y = y_temp.iloc[window_size-1:].values
        try:
            x_temp = x
            x = x_temp[window_size-1:]
        except Exception as e:
            x = None
    return x, y

def find_bulk_val(x, y, guess = None, fun = None, is_debug = False):
    """Find Value in bulk"""
    if guess is None:
        guess = [np.mean(y), 1, -1]
    if fun is None:
        fun  = funs._exponential_fit

    gm = GeneralModel(x, y, fun = fun)
    gm.fit(guess = guess, verbose = False, maxfev = np.int(1e6), do_plot = False)
    x_ax = np.arange(np.min(x), np.max(x)*10 , 1)
    y_hat = gm.predict(x_ax)
    ## You can also pick value that is lower than some threshold, but picking simply
    ## value that is far should do the job equally well.
    # try:
    #     idx = np.argmax(np.diff(y_hat) < (np.max(y) - np.min(y)) * 1e-3)
    #     y_bulk = y_hat[idx]
    # except Exception as e:
    #     y_bulk = y_hat[-1]
    if is_debug:
        plots.plot_generic([x, x_ax], [y, y_hat],
                            legend = ["dpoints", "fit"],
                            fmts = ["ro", "k--"],
                            scale = "logx")
        print("Bulk val is: ")
        print(y_hat[-1])
    return y_hat[-1]


def get_descriptor_fun(descriptor):
    """Fetch function based on its name"""
    if descriptor == "trim_mean":
        def trim_mean_(x): # get default for proportion to trim on both sideds
            return trim_mean(x, 0.1)
        fun_ = trim_mean_
    else:
        try:
            fun_ = getattr(np, descriptor)
        except Exception as e:
            print("Decriptor {} not found, selecting 'mean'.")
            fun_ = np.mean
    return fun_

def get_fit_params(As, Bs, force_positive = False):
    """Adjust parameters of fit such that they yield nonnegative value at x = 1

    Adjustment is done within the margin of one standard deviation.
    """
    a = np.mean(As)
    b = np.mean(Bs)
    if force_positive:
        astd = np.std(As)
        bstd = np.std(Bs)
        diff = 1.0 * a + b # should be zero
        sign = 1 if diff < 0 else -1
        # Option 1
        diff = np.minimum(np.abs(diff), bstd)
        b = b + sign * diff
        # Option 2
        # adj = np.minimum(astd, np.abs(diff))
        # a = (diff + sign*adj) - b
    return a, b

def is_null(value):
    ret = False
    if not value:
        ret = True
    elif np.isnan(value):
        ret = True
    # elif value.size() == 0:
    #     ret = True
    return ret

def smooth_outliers(x, qs = (0.05, 0.95), window_size = 5):
    """Replace single index outliers by mean of suroinding values"""
    x_qs = np.quantile(x, qs)
    loc = np.nonzero(np.logical_or(x < x_qs[0], x > x_qs[-1]))[0]
    # _, x_roll = rolling_mean(None, x, window_size)

    for i in loc: # could alsio make mask, but quick anyway
        try:
            # opt 1
            mask = list((i-1, i+1))
            x[i] = np.mean(x[mask])

            # opt 2
            # mask = list((i-2, min(i+2, x.shape[0] - 1)))
            # x[i-1:min(i+2, x.shape[0] - 1)] = np.mean(x[mask])
        except Exception as e:
            pass
    return x
