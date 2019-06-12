import numpy as np
from copy import deepcopy
import timeit
from inspect import signature
from scipy.optimize import curve_fit

from .model import Model

class GeneralModel(Model):
    """Simple Model of general y=f(x) functional relationship"""
    def __init__(self, x, y, fun):
        self.x = x
        self.y = y
        self.fun = fun
        super(GeneralModel, self).__init__()

    @property
    def x(self):
        """X"""
        return self._x
    @x.setter
    def x(self, value):
        self._x =  value

    @property
    def y(self):
        """Y"""
        return self._y
    @y.setter
    def y(self, value):
        self._y =  value

    def plot(self, fname = None):
        """Plot data

        Parameters
        --------
        fname: str
        """
        plots.plot_generic([self.x], [self.y], None, None, "", fname)

    def _fit_wrapper(self):
        """"Fit convenience wrapper"""
        fun = self.fun
        try:
            guess = [1] * len(signature(fun).parameters)
        except Exception as e:
            guess = None

        return fun, guess
