from sicm import io, analysis, plots, filters
from sicm.utils import utils

import numpy as np
import timeit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy

class Scan(object):
    """"""
    def __init__(self):
        self.raw_result = result
        self.exp_name = exp_name
        self.date = date
        self.result, self.idxs = self.downsample_to_linenumber()
        self.sel = sel

    def downsample_to_linenumber(self):
        uniqs, cnts = np.unique(self.raw_result["LineNumber"], return_counts=True)
        linenos = np.arange(5, max(uniqs), 3)
        result, idxs = io.downsample_to_linenumber(self.raw_result, linenos)
        return(result, idxs)

    def analyze_hopping_scan(self):
        """"""
        sel = self.sel
        if sel is None:
            sel = np.arange(0, len(result["LineNumber"])//1)
        plots.plot_hopping_scan(self.result, sel, self.exp_name, self.date)

    def analyze_approach(self):
        """"""
        #TODO: continue making the code cleaner, be fast though!

class LockIn(object):
    """LockIn class"""
    def __init__(self, datadir, file, chunk = 0):
        self.datadir = datadir
        self.fname = file
        self.chunk = chunk
        self.R_out = 50 # Resistance on output, 50 Ohm by default

    def load_data(self):
        """Load chunk from fname from datadir
        """
        data, date = io.load_data_lockin(self.datadir, self.fname, self.chunk)
        self.data = data
        self.date = date

    def plot(self, keypairs = None, xlog = False):
        """Default plot for LockIn
        """
        if keypairs is None:
            keypairs = list(zip(2*["frequency"], ["r", "phase"]))
        elif isinstance(keypairs, (list, )) and not isinstance(keypairs[0], (tuple, list)):
            keypairs = list(zip(len(keypairs)*["frequency"], keypairs))

        _name  = self.datadir.split("\\")[-1]
        plots.plot_lockin(  self.data, date = self.date, name = _name,
                            keys = keypairs, xlog = xlog)

    def trim_to_freq(self, frange = [0, -1]):
        """Trim data to some frequency range
        """
        self.data_full = copy.deepcopy(self.data)
        this_data = copy.deepcopy(self.data)
        f = self.data["frequency"]
        if frange[-1] == -1: frange[-1] = np.max(f)

        # Trim the data
        keep_id = np.logical_and(f > frange[0], f <= frange[-1])
        for k, v in copy.deepcopy(self.data).items():
            this_data[k] = v[keep_id]

        self.data = this_data
        if frange[-1] != np.max(f):
            assert len(f) > len(self.data["frequency"])

    def plot_fit(self, xlog= False, double_ax = False, plot_range = None):
        """Plots results of function fitting

        Calculates error of the fit as sum of square errors, scaled by maximum
        amplitude in sampled data.
        """
        x = self.data["frequency"].flatten()
        if plot_range:
            keep_id = np.logical_and(x > plot_range[0], x <= plot_range[-1])
        else:
            keep_id = np.asarray([True]*len(x))
        y_hat = self.fun(x, *self.popt) # evaluate fun at popt
        # Relative squared error
        sqerr = np.sum(np.power((y_hat - self.y)/np.max(self.y), 2))

        plt.style.use("seaborn")
        fig, ax = plt.subplots(figsize = (6.4, 4.8))
        text = "Internal Impedance fit @ {}".format(self.date)
        fig.suptitle(text, size  = 16, y = 0.96)

        # Plot amplitude on first axis
        ax.plot(x[keep_id], self.y[keep_id],
                label = r"$Z$ (data)", marker = ".", color = "k", alpha = 0.3,
                markevery = 1, linestyle = "")
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel(r"|Z| [$\Omega$]")
        if xlog:
            ax.set_xscale("log")
            ax.set_xlabel("log f [Hz]")
        # Add info box
        txt = "$error_{{fit}}$: {:.2E}".format(sqerr)
        plt.text(1.1, 0.1, txt, transform=ax.transAxes,
            fontdict = {'color': 'k', 'fontsize': 12, 'ha': 'left', 'va': 'center',
                        'bbox': dict(boxstyle="square", fc="w", ec="k", pad=0.2)})

        # Plot fit on second axis
        if double_ax:
            ax2 = ax.twinx()
        else:
            ax2 = ax
        ax2.plot(x[keep_id], y_hat[keep_id], "r--", label = r"$Z_{hat}$ (fit)", alpha = 0.7)

        if double_ax:
            ax2.set_ylabel(r"Z [$\Omega$]", color="r")
            ax2.tick_params("y", colors="r")
            ax2.grid(axis = "y", color = "r", alpha = .3, linewidth = .5, linestyle = ":")
            h2, l2 = ax2.get_legend_handles_labels()
        else:
            h2, l2 = ([], [])
        # Combine legends and show.
        h1, l1 = ax.get_legend_handles_labels()

        ax.legend(h1+h2, l1+l2, bbox_to_anchor = (1.3, 1.1), frameon = True)
        # utils.save_fig(text)
        plt.show()

    def _characterstic_fun4(f, *params):
        """Characterstic function of the instrument
        V_in/V_out = Z_in / (Z_in + R_out + Z)

        1/|Z_in| = |Y_in| = sqrt(1/R**2 + (2pi*f*C)**2)
        """
        R, Ca, Cb, L = params
        w = 2 * np.pi * f
        # impedance/admitance
        Z_in = (((1/(Ca+Cb)) + w*R*(Ca/(Ca+Cb)) + (w**2)*L*(Ca/(Ca+Cb))) / \
                (1 + w*R*((Ca*Cb)/(Ca+Cb)) + (w**2)*L*((Ca*Cb)/(Ca+Cb))))*(1/w)
        y = 1/ Z_in # admitance
        y = Z_in # impedance
        return y

    def _characterstic_fun4b(f, *params):
        """another fun"""
        z0, A, xc, fwhm = params
        # z0 ~ 0, xc~80k, fwhm~2e5, A = 471e9
        w = 2 * np.pi * f

        Z_in = z0 + (2*A/np.pi * (fwhm / (4*((w - xc)**2) + (fwhm**2))))
        y = Z_in
        return y

    def _characterstic_fun3(f, *params):
        """another fun"""
        R, C, L = params
        w = 2 * np.pi * f
        # Z_in = (R + w*L) / (1 + w*R*C + (w**2)*L*C)
        # Z_in = (R + w*L + (w**2)*R*L*C) / (1 + w*R*C)
        Z_in = w * (L / (1 + w*(L/R) + (w**2)*((L*C)/R)))
        y = Z_in
        return y

    def _characterstic_fun2(f, *params):
        """another fun"""

        R, C = params
        w = 2 * np.pi * f
        Y_in = np.sqrt((1/(R**2)) + (w*C)**2)# admitance
        y = 1 / Y_in
        return y


    def get_internal_impedance( self, Vout_peak = .1,
                                guess = np.array([1e6, 20e-12]), Z = 75):
        """Fit Internal impedance model"""

        self.Z = Z
        self.V_out = Vout_peak / np.sqrt(2)
        if not isinstance(guess, (np.ndarray, )): guess = np.asarray(guess)
        self.guess = guess

        if len(guess) == 4:
            # fun = LockIn._characterstic_fun4b
            fun = analysis.Fitter.lorentzian_fun
        elif len(guess) == 3:
            fun = LockIn._characterstic_fun3
        elif len(guess) == 2:
            fun = LockIn._characterstic_fun2
        self.fun = fun

        popt, y = self._get_internal_impedance(fun)
        self.popt = popt
        self.y = y

    def _get_internal_impedance(self, fun):
        """Fits model for internal impedance"""

        # Make sure (1-V_in/V_out) is positive value
        V_in = self.data["r"] # - (sorted(self.data["r"])[-1] - self.V_out)
        vv = V_in/self.V_out
        y = ((vv*(self.R_out + self.Z))/(1 - vv))
        # or equivalently
        # y = ((self.V_out / V_in) - 1) / ((self.R_out + self.Z))

        x = self.data["frequency"].flatten()

        start_time = timeit.default_timer()
        print("Fitting function to {} datapoints ...".format(len(x)))
        popt, _ = curve_fit(fun, x.flatten(), y.flatten(),
                            p0 = self.guess, maxfev = np.int(1e7),
                            # bounds = ([1, 1e-18, 1e-18, -1e3], [1e12, 1e4, 1e4, 1e3]),
                            # bounds = ([1., 1e-21, -1e18], [1e12, 1e3, 1e3]),
                            method = "dogbox")
        end_time = timeit.default_timer()
        print("Found parameters: {}.".format(popt))
        print("Finished in {:.3f} s".format(end_time - start_time))

        return popt, y

    def predict(self, f):
        """make prediction"""
        y_hat = self.fun(f, *self.popt)
        return y_hat
