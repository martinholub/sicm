import numpy as np

def _exponential_fit(f, *params):
    """Fit general exponential relationship
    Parameters
    -----------
    f: array-like
        Relative current
    params: tuple of floats
        Parameters of the model

    # guess = 2.4e-5, 247.8, 140
    """
    A, B, C = params
    T = A * np.exp(B / (f - C))
    return T

def _logarithmic_fit(f, *params):
    """for polyfit
    # guess = 1e-5, 1e2
    """

    A, B = params
    T = A * np.log(f) + B
    return T

def _hyperbolic_fit(x, *params):
    """hyperbolic fit
    """
    A, B, C = params
    y = A / (x - C) + B
    return y

def _ratio_fit(x, *params):
    """hyperbolic fit
    """
    A, B, C = params
    y = A * (x + B) / (x + C)
    return y

def _linear_fit(x, *params):
    """Fit linear realtionship"""
    a, b = params
    y = a*x + b
    return y
