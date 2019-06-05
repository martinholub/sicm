import os
import numpy as np
import pandas as pd
import timeit
from scipy.optimize import curve_fit


from sicm import io
from sicm.plots import plots
from sicm.utils import utils
from .model import Model

import matplotlib.pyplot as plt

class TemperatureBulkModel(object):
    def __init__(self):
        pass

class HotSpotModel(object):
    def __init__(self):
        pass

class TemperatureModel(Model):
    """Model of current-distance-temperature dependence"""
    def __init__(self, x, y, T, T0 = 298.15):
        self.T0 = T0
        self.y = y
        self.x = x
        self.T = T
        super(TemperatureModel, self).__init__()

    @property
    def y(self):
        """Current relative to the bulk value"""
        return self._y
    @y.setter
    def y(self, value):
        # Assure current is being passed as relative.
        assert np.max(value) >= 1, "Current must be passed in as relative."
        self._y =  value

    @property
    def x(self):
        """Separation From Surface"""
        return self._x
    @x.setter
    def x(self, value):
        assert np.max(value) >= 1, "Distance must be passed as relative."
        self._x =  value

    @property
    def T(self):
        """Temperature"""
        return self._T
    @T.setter
    def T(self, value):
        if isinstance(value, (np.int, np.float)):
            value = self._model_temperature(value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            assert np.max(value) < 100, "Temperature must be passed as difference w.r.t bulk."
        self._T =  value

    def _model_temperature(self, T_sub):
        """Apply model of temperature"""
        T0 = self.T0
        x = self.x
        try:
            r_sub = self.r_sub
        except Exception as e:
            r_sub = np.min(x)

        A = (T_sub - T0) * r_sub
        T = A / x
        return T

    def plot(self, fname = None):
        """Plot data

        Parameters
        --------
        """
        x = self.x
        y = self.y / self.T
        leg = ""
        plots.plot_generic([x], [y], [r"$z/d$"], [r"$\frac{I}{I_{bulk}}/\Delta T$"], leg, fname)

    def _fit_wrapper(self):
        """"Fit convenience wrapper"""
        guess = 2.4e-5, 247.8, 140
        def _exponential_fit(f, *params):
            """Fit general exponential relationship
            Parameters
            -----------
            f: array-like
                Relative current
            params: tuple of floats
                Parameters of the model
            """
            A, B, C = params
            T = A * np.exp(B / (f - C))

            # for polyfit
            # A, B = params
            # T = A * np.log(f) + B
            return T
        return _exponential_fit, guess

    def fit(self, guess = None, double_ax = True):
        fun, guess_ = self._fit_wrapper()
        x = self.x # relative current
        y = self.y #/ np.min(self.y) # Temperature
        if guess is None:
            guess = guess_
        #Solve non-linear lsq problem, yielding parama minimizing fun(x,*params)-y
        start_time = timeit.default_timer()
        try:
            print("Fitting {} to {} datapoints ...".format(fun.__qualname__, len(x)))
        except AttributeError as e:
            print("Fitting {} to {} datapoints ...".format(fun.__name__, len(x)))

        popt, _ = curve_fit(fun, x, y, p0 = guess, maxfev = np.int(1e6),
                            method = "lm")

        end_time = timeit.default_timer()
        print("Found parameters: {}.".format(popt))
        print("Finished in {:.3f} s".format(end_time - start_time))
        self.popt = popt

        self.plot_fit(y, x, double_ax = double_ax)
