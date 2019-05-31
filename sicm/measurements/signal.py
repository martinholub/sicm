import numpy as np
from copy import deepcopy
import os

import matplotlib.pyplot as plt

from scipy.signal import detrend, welch, periodogram
from scipy import stats

from sicm.utils import utils
from sicm.plots import plots
from sicm.filters import LowPassButter

class Signal(object):
    def __init__(self, x, y, datadir = None, exp_name = None):
        self.x = x
        self.y = y
        self.datadir = datadir
        self.exp_name = exp_name

    @property
    def exp_name(self):
        return self._exp_name
    @exp_name.setter
    def exp_name(self, value):
        if not isinstance(value, (str, )) and isinstance(value, (list, tuple)):
            value = "-".join(value)
        self._exp_name = value

    def get_fpath(self):
        """Make filename"""
        try:
            fname = os.path.join(self.datadir, self.exp_name)
        except TypeError as e:
            fname = None
        return fname

    def detrend_signal(self, do_plot = True):
        """ Detrends Data

        References
        ------------
        [1]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
        [2]: https://gist.github.com/junzis/e06eca03747fc194e322
        """
        x = deepcopy(self.x); y = deepcopy(self.y)
        # Obtain main trend from data with low pass filter,
        lpb = LowPassButter()
        y_filt = lpb.filter(y, cutoff = .5, fs = 10, order = 3)
        # Fit stright line to the the trend, remove artifacts from beginning
        cutid = len(y)//10
        x_ = x - x[0] # start from zero on time axis
        m, b, _, _, _ = stats.linregress(x_[cutid:], y_filt[cutid:])
        trend = m*x_ + b
        # substratct the trend from data.
        ret = y - (m*x_) # -b
        # ret = detrend(sig, type = "linear")
        # Show diagnostic plot.
        if do_plot:
            self.plot_detrend_diagnose(y_filt, trend, ret, x_, cutid)
        self.detrend = ret
        return ret

    def plot_detrend_diagnose(self, filt, trend, ret, x, cutid = 100):
        """Plot intermediate steps of detrending process

        If it looks like detrending is not doing what it is supposed to, you should
        adjust values of cutoff, fs and order parameters.
        """
        orig = deepcopy(self.y)

        plt.style.use("seaborn")
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (2*6.4, 4.8))
        axs.flatten()
        fig.suptitle("Detrending Diagnostic Plot", size  = 16, y = 0.96)
        for y, fmt, l in zip( (orig, filt, trend, ret),
                            ("b-", "k-", "g--", "r-"),
                            ("original", "filtered", "trend", "final")):
            axs[0].plot(x[0:cutid], y[0:cutid], fmt, alpha = 0.2)
            axs[0].plot(x[cutid:], y[cutid:], fmt, alpha = 0.5, label = l)
            axs[1].plot(x[cutid:], y[cutid:], fmt, alpha = 0.5, label = l)
        axs[0].axvline(x[cutid], color = "gray", linestyle = ":", label = "cutid")
        axs[0].set_title("Full range")
        axs[1].set_title("Detail after cutid")
        for ax in axs:
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(r"$\theta$ [$\degree$]")
            ax.legend()

    def plot(self, x_lab = "x", y_lab = "y", legend = None, **kwargs):
        fname = self.get_fpath()
        try:
            avg = np.mean(self.y)
            std = np.std(self.y)
            if legend is None and legend != "":
                legend = "$\mu$: {:.3E}\n$\sigma$: {:.3E}".format(avg, std)
        except Exception as e:
            pass

        x_plot = self.x if isinstance(self.x, (list, )) else [self.x]
        y_plot = self.y if isinstance(self.y, (list, )) else [self.y]
        plots.plot_generic(x_plot, y_plot, [x_lab], [y_lab], legend, fname, **kwargs)

    def analyze(self, range = None, what = "noise", fpath = None):
        """Analyze data

        Data should be suplied as x,y pair to the class constructor.

        Parameters
        ------------
        range: array-like
            Sequence of length 2, indicating section of data to use. If float,
            assumed to be given in data values, if int, assumed to be indices to array.

        Returns:
        --------

        """
        if range is not None:
            assert isinstance(range, (list, tuple, np.ndarray))
            assert len(range) == 2

            if isinstance(range[0], (float)):
                keep_id = np.logical_and(self.x > range[0], self.x <= range[-1])
                yy = np.squeeze(np.asarray(self.y)[keep_id])
                xx = self.x[keep_id]
            elif isinstance(range[0], (int)):
                yy = np.asarray(self.y)[range[0]:range[-1]]
                xx = self.x[range[0]:range[-1]]
            else:
                raise ValueError("Supplied range must be either int or float.")
        else:
            yy = deepcopy(self.y)
            xx = deepcopy(self.x)

        if what == "noise":
            self._get_noise_level(xx, yy, fpath)
        if what == "psd":
            self._get_psd(xx, yy, fpath)

    def _get_noise_level(self, xx, yy, fpath):
        """Obtain noise level from data

        Noise level is quantified as standard deviation
        `(std=sqrt(mean(abs(x - mean(x)**2)))`.

        Parameters
        ------------
        xx: array-like
        yy: array-like

        Returns:
        --------
        noise: float
            Noise level, in same units as data.
        """
        noise = np.std(yy)
        if np.max(np.abs(yy)) < 1.0:
            leg = "Noise level <x>: {:.3f} pA".format(noise*1e12)
            y_lab = "Current [nA]"
            yy *= 1e9
        else:
            leg = r"Noise level <x>: {:.3f}$\degree$".format(noise)
            y_lab = r"Phase [$\degree$]"

        if fpath is not None:
            fpath += "_noise"

        plots.plot_generic([xx], [yy], ["time [s]"], [y_lab], leg,
                            fpath)
        self.noise = noise

    def _get_psd(self, xx, yy, fpath):
        """Obtain and plot power spectral density

        Apply PSD analysis to timeseries (xx, yy). This function is used to estimate
        noise RMS and its frequency. Should be adjusted for different purposes.

        Parameters
        ------------
        xx: array-like
        yy: array-like

        Returns:
        --------
        noise: float
            Noise level, in same units as data.
        """
        fs = 1 / np.diff(xx)[0]
        # adding: nfft = 512 or noverlap = 128 does not change much
        # same goes for using different windows
        f, Pyy_spec = welch(yy, fs, window = "hamming", nperseg = 256,
                            detrend = "constant", scaling = "spectrum",
                            return_onesided = True)

        max_val = np.max(np.sqrt(Pyy_spec))
        max_f = f[np.argmax(Pyy_spec)]
        if np.max(np.abs(yy)) < 1.0:
            leg = "Noise RMS: {:.3f} pA @ f: {:.3f} Hz".format(max_val*1e12, max_f)
            y_lab = "log Spectrum [A RMS]"
        else:
            leg = r"Noise RMS: {:.3f}$\degree$ @ f: {:.3f} Hz".format(max_val, max_f)
            y_lab = r"log Spectrum [$\degree$ RMS]"

        if fpath is not None:
            fpath += "_psd"

        plots.plot_generic([f], [(np.sqrt(Pyy_spec))], ["f [Hz]"],
                            [y_lab], leg, fpath)
        self.noise = np.sqrt(Pyy_spec)
