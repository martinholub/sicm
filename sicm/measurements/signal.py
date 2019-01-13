import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import detrend
from scipy import stats

from sicm.utils import utils
from sicm import plots
from sicm.filters import LowPassButter

class Signal(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def detrend_signal(self, do_plot = True):
        """ Detrends Data

        References
        ------------
        [1]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
        [2]: https://gist.github.com/junzis/e06eca03747fc194e322
        """
        x = self.x; y = self.y
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
        orig = self.y

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

    def get_noise_level(self, range = None):
        """Obtain noise level from data

        Data should be suplied as x,y pair to the class constructor. Noise level is
        quantified as standard deviation `(std=sqrt(mean(abs(x - mean(x)**2)))`.

        Parameters
        ------------
        range: array-like
            Sequence of length 2, indicating section of data to use. If float,
            assumed to be given in data values, if int, assumed to be indices to array.

        Returns:
        --------
        noise: float
            Noise level, in same units as data.

        """
        if range is not None:
            assert isinstance(range, (list, tuple, np.ndarray))
            assert len(range) == 2

            if isinstance(range[0], (float)):
                keep_id = np.logical_and(self.x > range[0], self.x <= range[-1])
                yy = np.asarray(self.y)[keep_id]
                xx = self.x[keep_id]
            elif isinstance(range[0], (int)):
                yy = np.asarray(self.y)[range[0]:range[-1]]
                xx = self.x[range[0]:range[-1]]
            else:
                raise ValueError("Supplied range must be either int or float.")
        else:
            yy = self.y
            xx = self.x

        noise = np.std(yy)
        leg = "Noise level <x>: {:.3f} pA".format(noise*1e12)
        plots.plot_generic([xx], [yy*1e9], ["time [s]"], ["Current [nA]"], leg,
                            "noise_level")
        self.noise = noise
