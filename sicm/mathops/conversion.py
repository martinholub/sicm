import numpy as np
import pandas as pd
import os
from sicm.utils import utils

def get_polyfun():
    def polyfun(x, params):
        return np.sum((p*x**i for i, p in enumerate(params[::-1])))
    return polyfun

def convert_measurements(data, fun = None, params = None):
    # Params should be a tuple, if not, load from file
    if params is None:
        try:
            pwd = os.getcwd().replace("\\notebooks", "")
            pth = os.path.join(pwd, "data/s9r003.json")
            d = utils.load_dict(pth)
            params = d["fit"]["coeff (mean)"]
        except Exception as e:
            print("You must supply parameters for the conversion!")
            raise e
    # Function is a simple polynomial
    if fun is None:
        fun = get_polyfun()
    # Apply function on flattened array and reshape back
    in_shape = data.shape
    val = fun(data.flatten(), params)
    val = np.reshape(val, in_shape)

    return val
