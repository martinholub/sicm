from sicm import io, analysis, filters
from sicm.plots import plots
from sicm.utils import utils
from .model import Model

import numpy as np
import timeit
from scipy.optimize import curve_fit

from skimage.morphology import square
from skimage.filters import rank, threshold_otsu

import matplotlib.pyplot as plt

class GaussianBeam(Model):
    """Fit a gaussian beam model to an image of laser spot
    """

    def __init__(self, datadir, exp_name):
        self.img = self._load_image(datadir, exp_name)
        super(GaussianBeam, self).__init__()

    @property
    def fun(self):
        val, _ = self._fit_wrapper()
        return val

    @property
    def fwhm(self):
        try:
            val = self.popt[1] * np.sqrt(2 * np.log(2))
        except Exception as e: #popt not assigned yet
            val = np.nan
        return val

    def _load_image(self, datadir, exp_name):
        self.data_dir = datadir
        self.exp_name = exp_name
        img = io.rgb2gray(io.load_image(datadir, exp_name))
        return img

    def _filt(self, size = 10):
        img_filt = rank.mean(self.img, square(size))
        return img_filt

    def thresh(self, filt_size = 10):
        self.img_filt = self._filt(filt_size)
        thresh = threshold_otsu(self.img_filt)
        img_binary = self.img_filt > thresh
        self.img_binary = img_binary

    def fit_circle(self):
        try:
            data = self.img_binary
        except Exception as e:
            self.thresh()
            data = self.img_binary
        mask = np.nonzero(data)
        y_c = np.int(np.mean(mask[0]))
        x_c = np.int(np.mean(mask[1]))

        fig, ax = plt.subplots(1, 1)
        ax.imshow(data, origin = )
        ax.plot(x_c, y_c, marker = "*", color = "red", markersize = 18,
                mew = 4)

    def _assign_extents(self, x_size, y_size):
        """TODO: Continue Here"""
        pass


    def _cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    def _pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)

    def _fit_wrapper(self):
        """"Fit convenience wrapper"""
        guess = [0.5, 50e-6]
        def _gaussian_beam_model(x, *params):
            """Fit gaussian beam model to an image of focal spot

            Parameters
            -----------
            x: array-like
                (M, N) or (M, N, 3) image data
            params: tuple of floats
                Parameters of the model
            """
            P, w0 = params
            y = (2 * P) / (np.pi * w0**2) * np.exp((-2 * x**2) / w0**2)
            return y
        return _gaussian_beam_model, guess

    def fit(self, *args, **kwargs):
        self._fit(*args, **kwargs)

    def plot(self, fname = None):
        """Plot data

        Parameters
        --------
        fname: str
            Optional fpath for saving data
        """
        y = self.y
        x = self.x
        plots.plot_generic( [x], [y], ["Driving Current [a.u.]"], ["Power [mw]"],
                            fname = fname)
