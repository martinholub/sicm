from sicm import io, analysis, filters
from sicm.plots import plots
from sicm.utils import utils
from .model import Model

import numpy as np
import timeit
from scipy.optimize import curve_fit
from copy import deepcopy

from skimage.morphology import square
from skimage.filters import rank, threshold_otsu

from image_processing import load_image_container

import matplotlib.pyplot as plt

class GaussianBeam(Model):
    """Fit a gaussian beam model to an image of laser spot

    Current implementation is not working because no raw_image available.
    Also, the camera is sautrated!

    Parameters
    ---------
    datadir: str
        Path to directory containing the image.
    exp_name: str
        Name of the image. Should be in `.tiff` format!
    is_debug: bool
        Should diagnostic plots be displayed?

    """

    def __init__(self, datadir, exp_name, is_debug = False):
        self.img = self._load_image(datadir, exp_name)
        self.is_debug = is_debug
        super(GaussianBeam, self).__init__()

    @property
    def fun(self):
        val, _ = self._fit_wrapper()
        return val

    @property
    def fwhm(self):
        """Full Widht at Half Maximum"""
        try:
            val = self.popt[1] * np.sqrt(2 * np.log(2))
        except Exception as e: #popt not assigned yet
            val = np.nan
        return val

    def _load_image(self, datadir, exp_name):
        """Load Image as GrayScale values

        TODO: Addapt this for raw tiffs once available
        """
        self.data_dir = datadir
        self.exp_name = exp_name
        try:
            # img = io.rgb2gray(io.load_image(datadir, exp_name))
        except Exception as e:
            img = np.random.randn(640, 480)
        return img

    def _filt(self, size = 10):
        """Filter the image with moving average filter

        References
        --------------
        [1]  https://scikit-image.org/docs/dev/api/skimage.filters.rank.html#skimage.filters.rank.mean
        """
        try:
            img_filt = rank.mean(self.img, square(size))
        except ValueError as e:
            # img_ = deepcopy(self.img) / self.img.max()
            # img_filt = rank.mean(img_, square(size))
            # img_filt = img_filt * self.img.max()
            img_filt = deepcopy(self.img)
        return img_filt

    def thresh(self, filt_size = 10):
        """Apply Otsu thresholding to obtain binary mask"""
        self.img_filt = self._filt(filt_size)
        thresh = threshold_otsu(self.img_filt)
        img_binary = self.img_filt > thresh
        self.img_binary = img_binary

    def fit_circle(self):
        """Fit circle to binary image

        This is naive approach, but works well for simple scenarios.
        """
        try:
            data = self.img_binary
        except Exception as e:
            self.thresh()
            data = self.img_binary
        mask = np.nonzero(data)
        self.y_c = np.int(np.mean(mask[0]))
        self.x_c = np.int(np.mean(mask[1]))

        if self.is_debug:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(data, origin = "upper")
            ax.plot(self.x_c, self.y_c, marker = "*", color = "red",
                    markersize = 18, mew = 4)

    def _assign_extents(self, x_extent, y_extent):
        """Assing X,Y image extents

        This basically yields pixel_size, assuming pixels are square. In future,
        you can use this infromation to simplify your aproach.

        Parameters
        ----------
        x_extent: float
            Horizontal size of FOV in meters.
        y_extent: float
            Vertical size of FOV in meters.

        Returns
        ----------
        x_ax: np.ndarray
        y_ax: np.ndarray

        """
        len_x = self.img.shape[1]
        len_y = self.img.shape[0]
        is_matching =   ((len_y >= len_x) and (y_extent >= x_extent)) or \
                        ((len_y < len_x) and (y_extent < x_extent))
        assert is_matching, "Coordinate order is inverted! Please switch extents."

        x_ax = np.linspace(0, x_extent, len_x)
        y_ax = np.linspace(0, y_extent, len_y)
        return x_ax, y_ax

    def _assign_pixel_size(self, pixel_size):
        """Create axis based on know pixel size"""
        len_x = self.img.shape[1]
        len_y = self.img.shape[0]
        x_ax = np.linspace(0, len_x * pixel_size, len_x)
        y_ax = np.linspace(0, len_y * pixel_size, len_y)
        return x_ax, y_ax

    def _shift_center(self, x_ax, y_ax):
        """Shift Center of axis to center of fitted circle

        TODO: Check that the cordinates match exactly (no +-1 offset)

        Returns
        ---------
        x_ax, y_ax: np.ndarray
            Axes in real-world coordinates, centered at circle center.
        """
        #x_ax = np.roll(x_ax, self.x_c)
        x_ax = x_ax - x_ax[self.x_c]
        # y_ax = np.roll(y_ax, self.y_c)
        y_ax = y_ax - y_ax[self.y_c]
        return x_ax, y_ax

    def _cart2pol(self, x, y):
        """Convert Cartesian to polar"""
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    def _convert_to_polar(self, x, y, signed = False):
        """Convert cartesian cooridnates to polar

        If signed, returns doublesided distribution.
        """
        xx, yy = np.meshgrid(x, y)
        rho, phi = self._cart2pol(xx, yy)
        if signed:
            sign = np.sign(phi)
            sign[sign == 0] = 1
            return rho * sign
        else:
            return rho
        return rho

    def convert_to_polar(self, pixel_size, extents = None, double_sided = False):
        """Convert image coordinates to real-world polar representation

        TODO: Offer possibility to trim data.

        Parameters
        ----------
        x_extent: float
            Horizontal size of FOV in meters.
        y_extent: float
            Vertical size of FOV in meters.
        double_sided: bool
            Return double-sided spectrum?
        """

        if extents:
            x_ax, y_ax = self._assign_extents(x_extent, y_extent)
        else:
            x_ax, y_ax = self._assign_pixel_size(pixel_size)
        x_ax, y_ax = self._shift_center(x_ax, y_ax)
        self.rho_ax = self._convert_to_polar(x_ax, y_ax, double_sided)

        if self.is_debug:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(self.rho_ax)
            ax.plot(self.x_c, self.y_c, marker = "*", color = "red",
                    markersize = 18, mew = 4)

        x = self.rho_ax.flatten()
        y = self.img_filt.flatten()
        self.x = x# x[x>2e-5]
        self.y = y# y[x>2e-5]

    def _fit_wrapper(self):
        """"Fit convenience wrapper"""
        guess = [1e-3, 50e-6]
        def _gaussian_beam_model(x, *params):
            """Fit gaussian beam model to an image of focal spot

            Parameters
            -----------
            x: array-like
                (M, N) or (M, N, 3) image data
            params: tuple of floats
                Parameters of the model

            References
            ------------
            [1]  https://www.rp-photonics.com/gaussian_beams.html
            """
            P, w0 = params
            y = (2 * P) / (np.pi * w0**2) * np.exp((-2 * x**2) / w0**2)
            return y

        return _gaussian_beam_model, guess

    def remove_saturated(self, sat_thresh = np.inf):
        keep = self.y < sat_thresh
        self.y = self.y[keep]
        self.x = self.x[keep]

    def remove_edges(self, edge_thresh = np.inf):
        keep = np.abs(self.x) < edge_thresh
        self.y = self.y[keep]
        self.x = self.x[keep]

    def fit_model(self, *args, **kwargs):
        self.fit(*args, **kwargs)

    def plot(self, fname = None):
        """Plot data

        Parameters
        --------
        fname: str
            Optional fpath for saving data
        """
        y = self.y
        x = self.x
        plots.plot_generic( [x], [y], ["Distance from center [m]"], ["Power [A.U.]"],
                            fname = fname)
