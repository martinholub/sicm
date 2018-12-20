from sicm import io, analysis, plots, filters
from sicm.utils import utils

import numpy as np
import timeit
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy
from math import ceil

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
        self.data, self.date = io.load_data_lockin(self.datadir, self.fname, self.chunk)

    def plot(self, keypairs = None, xlog = False):
        """Default plot for LockIn

        Parameters
        ---------
        xlog: bool
            Should x axis be on logscale?
        keypairs: list of tuples
            List of the form [(x1, y1), (x2, y2), ...] where the pairs in tuples
            are keys in data to be used for x and y axes respectivelly. If only
            simple list is provided (e.g. [y1, y2]), then all xs are assumes to
            be 'frequency'.
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

        Parameters
        -----------
        frange: array-like
            Bounds to trim to as values of [low, high] frequency.
        """
        # Make sure you always trim from full dataset
        if not hasattr(self, "data_full"):
            self.data_full = copy.deepcopy(self.data)
        else:
            self.data = copy.deepcopy(self.data_full)
        this_data = copy.deepcopy(self.data)

        #Get the range of frequencies we trim to
        if not frange: frange = [0, -1]
        if isinstance(frange, (tuple, )): frange = list(frange) # make mutable
        f = self.data["frequency"]
        if frange[-1] == -1: frange[-1] = np.max(f)

        # Trim the data
        keep_id = np.logical_and(f >= frange[0], f <= frange[-1])
        for k, v in copy.deepcopy(self.data).items():
            this_data[k] = v[keep_id]

        # Assign and check that data has been trimmed
        self.data = this_data
        if frange[-1] != np.max(f):
            assert len(f) > len(self.data["frequency"])

    def plot_fit(self, xlog= False, double_ax = False, plot_range = None):
        """Plots results of function fitting

        Calculates error of the fit as sum of squared errors, scaled by maximum
        amplitude in sampled data.

        Parameters
        -----------
        xlog: bool
            Should the x axis be on log scale?
        double_ax: bool
            Should the second line be plotted on its own yaxis?
        plot_range: array-like
            Bounds to trim to as values of [low, high] frequency.
        """

        x = self.data["frequency"].flatten()
        if plot_range:
            keep_id = np.logical_and(x > plot_range[0], x <= plot_range[-1])
        else:
            keep_id = np.asarray([True]*len(x))
        y_hat = self.fun(x, *self.popt) # evaluate fun at popt
        # Relative squared error
        sqerr = np.sum(np.power((y_hat - self.y)/np.max(self.y), 2))
        # Apply correction
        y_corr = analysis.Fitter.apply_correction(self, f = x, r_e = y_hat)

        plt.style.use("seaborn")
        fig, ax = plt.subplots(figsize = (6.4, 4.8))
        text = "Internal Impedance Fit @ {}".format(self.date)
        fig.suptitle(text, size  = 16, y = 0.96)

        # Plot amplitude on first axis
        ax.plot(x[keep_id], self.y[keep_id],
                label = r"$Z$ (data)", marker = ".", color = "k", alpha = 0.3,
                markevery = 1, linestyle = "")
        ax.plot(x[keep_id], self.r_m[keep_id], "g-", label = r"$Z_{in} (corr)$", alpha = 0.7)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel(r"|Z| [$\Omega$]")
        if xlog:
            ax.set_xscale("log")
            ax.set_xlabel("log f [Hz]")
        # Add info box
        txt = "$f_0$: {:.2f} Hz\n$Q$: {:.2f}\n$error_{{fit}}$: {:.2E}".format(\
                        self.popt[2]/(2*np.pi), self.popt[1], sqerr)
        plt.text(1.1, 0.1, txt, transform=ax.transAxes,
            fontdict = {'color': 'k', 'fontsize': 12, 'ha': 'left', 'va': 'center',
                        'bbox': dict(boxstyle="square", fc="w", ec="k", pad=0.2)})

        # Plot fit on second axis
        ax2 = ax.twinx() if double_ax else ax
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

    def _get_characteristic_fun(self, num_params):
        """Return characteristic function based on number of parameters"""

        def _characterstic_fun4(f, *params):
            """Model of a resonator

            References:
              https://www.qsl.net/in3otd/electronics/Z_fitting/impedance_fitting.html
            """
            R, Ca, Cb, L = params
            w = 2 * np.pi * f
            # impedance/admitance
            Z_in = (((1/(Ca+Cb)) + w*R*(Ca/(Ca+Cb)) + (w**2)*L*(Ca/(Ca+Cb))) / \
                    (1 + w*R*((Ca*Cb)/(Ca+Cb)) + (w**2)*L*((Ca*Cb)/(Ca+Cb))))*(1/w)
            y = Z_in
            return y

        def _characterstic_fun4b(f, *params):
            """Lorentzian function

            References:
              https://www.originlab.com/doc/Origin-Help/Lorentz-FitFunc
             """
            #Some good starting values: z0 ~ 0, xc~80k, fwhm~2e5, A = 471e9
            z0, A, xc, fwhm = params
            w = 2 * np.pi * f
            # impedance
            Z_in = z0 + (2*A/np.pi * (fwhm / (4*((w - xc)**2) + (fwhm**2))))
            y = Z_in
            return y

        def _characterstic_fun3(f, *params):
            """Variations of RLC circuit

            References:
              https://www.qsl.net/in3otd/electronics/Z_fitting/impedance_fitting.html
             """
            R, C, L = params
            w = 2 * np.pi * f
            # impedance
            # Z_in = (R + w*L) / (1 + w*R*C + (w**2)*L*C) # Circuit B
            # Z_in = (R + w*L + (w**2)*R*L*C) / (1 + w*R*C) # circuit C
            Z_in = w * (L / (1 + w*(L/R) + (w**2)*((L*C)/R))) # Circuit A
            y = Z_in
            return y

        def _characterstic_fun2(f, *params):
            """RC Circuit, very simplfied model of LockIn

            1/|Z_in| = |Y_in| = sqrt(1/R**2 + (2pi*f*C)**2)
            """
            R, C = params
            w = 2 * np.pi * f
            # admitance
            Y_in = np.sqrt((1/(R**2)) + (w*C)**2)
            y = 1 / Y_in # impedance
            return y

        if num_params == 4:
            fun = analysis.Fitter.lorentzian_fun
        elif len(guess) == 3:
            fun = _characterstic_fun3
        elif len(guess) == 2:
            fun = _characterstic_fun2
        return fun

    def get_internal_impedance( self, Vout_peak = .1,
                                guess = np.array([1e6, 20e-12]), Z = 75):
        """Fit model for internal impedance of lockin amplifier

        References
        --------------
          https://www.uq.edu.au/_School_Science_Lessons/30.5.6.2.GIF
          http://mlg.eng.cam.ac.uk/mchutchon/ResonantCircuits.pdf
          https://keisan.casio.com/exec/system/1258032649
        """
        # Assignments
        self.Z = Z
        self.V_out = Vout_peak / np.sqrt(2)
        if not isinstance(guess, (np.ndarray, )): guess = np.asarray(guess)
        self.guess = guess
        # Decide which function are we using
        self.fun = self._get_characteristic_fun(len(guess))
        # Call fitting procedure
        self.popt, self.y = self._get_internal_impedance(self.fun)

    def _get_internal_impedance(self, fun):
        """Fits model for internal impedance

        Things go wrong if `max(V_in)>Vout_peak/sqrt(2)`. For this reason,
        I pass `Vout_peak=ceil(max(V_in)*sqrt(2)), 5)`, which is probably ok and
        within error of the measurment.

        Parameters
        -----------
        fun: callable
            Function fo the form `y = f(x, *params)`, to be fitted
        Returns
        ---------
        popt: tuple
            Parameters of the fitted function
        y: array-like
            Experimental data used for fitting
        """
        # Make sure (1-V_in/V_out) is positive, needed if passing unadjusted Vout_peak
        V_in = self.data["r"] #-(sorted(self.data["r"])[-1] - self.V_out)
        # Internal Impedancefrom:  V_in/V_out = Z_in / (Z_in + R_out + Z)
        vv = V_in/self.V_out
        y = ((vv*(self.R_out + self.Z))/(1 - vv))
        # or equivalently
        # y = ((self.V_out / V_in) - 1) / ((self.R_out + self.Z))

        x = self.data["frequency"].flatten()
        # Do the fitting
        start_time = timeit.default_timer()
        print("Fitting function to {} datapoints ...".format(len(x)))
        popt, _ = curve_fit(fun, x.flatten(), y.flatten(),
                            p0 = self.guess, maxfev = np.int(1e7),
                            bounds = ([0, 0, 0, -1e3], [1e9, 1e4, 1e9, 1e3]),
                            method = "trf")
        end_time = timeit.default_timer()
        print("Found parameters: {}.".format(popt))
        print("Finished in {:.3f} s".format(end_time - start_time))

        return popt, y

    def predict(self, f):
        """Evaluate fitted function at frequency f
        """
        y_hat = self.fun(f, *self.popt)
        # Apply correction
        y_corr = analysis.Fitter.apply_correction(self, f = f, r_e = y_hat)
        return y_corr

    #### Stuff for oscillator
    def plot_fit_osc(self, data, xlog= False, double_ax = False, plot_range = None):
        """Plots results of function fitting

        Calculates error of the fit as sum of squared errors, scaled by maximum
        amplitude in sampled data.

        TODO: Make this one function with `plot_fit`

        Parameters
        -----------
        xlog: bool
            Should the x axis be on log scale?
        double_ax: bool
            Should the second line be plotted on its own yaxis?
        plot_range: array-like
            Bounds to trim to as values of [low, high] frequency.
        """
        x = data["x"]
        popt = data["popt"]
        y = data["y"]
        y_hat = data["y_hat"]
        y_hat_corr = data["y_hat_corr"]

        if plot_range:
            keep_id = np.logical_and(x > plot_range[0], x <= plot_range[-1])
        else:
            keep_id = np.asarray([True]*len(x))

        # Relative squared error
        try:
            sqerr = np.sum(np.power((y_hat[keep_id] - y[keep_id])/np.max(y[keep_id]), 2))
        except Exception as e:
            sqerr = np.nan

        plt.style.use("seaborn")
        fig, ax = plt.subplots(figsize = (6.4, 4.8))
        text = "Oscillator Model Fit @ {}".format(self.date)
        fig.suptitle(text, size  = 16, y = 0.96)

        # Plot amplitude on first axis
        try:
            ax.plot(x[keep_id], y[keep_id],
                    label = r"$Z$ (data)", marker = ".", color = "k", alpha = 0.3,
                    markevery = 1, linestyle = "")
        except Exception as e:
            pass # used in debugging

        if y_hat_corr is not None:
            ax.plot(x[keep_id], y_hat_corr[keep_id], "g-",
                    label = r"$Z_{in} (corr)$", alpha = 0.7)
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
        ax2 = ax.twinx() if double_ax else ax
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

    def _resonator_model(f, *params):
        """Model of a resonator

        Cant make this one fit anything

        References:
          https://www.qsl.net/in3otd/electronics/Z_fitting/impedance_fitting.html
        """
        R, Ca, Cb, L = params
        w = 2 * np.pi * f
        # impedance/admitance
        Z_in = (((1/(Ca+Cb)) + w*R*(Ca/(Ca+Cb)) + (w**2)*L*(Ca/(Ca+Cb))) / \
                (1 + w*R*((Ca*Cb)/(Ca+Cb)) + (w**2)*L*((Ca*Cb)/(Ca+Cb))))*(1/w)
        y = Z_in
        return y

    def characterize_oscillator(self, osc_data, guess):
        """"Describe fork in terms of its parameters

        Fit Lorentzian function to the experimental impedance data. Then try to reconstruct
        R, Cs, Cp, L parameters of BvD equivalent circuit for resonator using
        relationships in Lee2007 [1]. This approach is probably valid, only
        keep in mind that you are fitting impedance and thus the dimensional
        parameter i0 in the fit has units of Ohm, and you need to treat it differently,
        possibly as resistance.


        Parameters
        ------------
        osc_data: dict
            Data of the frequency sweep for the resonator. Must have keys 'r' and
            'frequency'.
        guess: array-like
            Initial guess for parameters (i0, Q, w0, CC).
            (4e5, 2.2, 596e3*2*np.pi, 1e-1) works well.

        Returns
        ------------
        osc_dict: dict
            Dict with keys x,y,y_hat,y_hat_corr,popt. Data for plotting.


        References
          https://doi.org/10.1063/1.2756125
        """

        fun = analysis.Fitter.lorentzian_fun
        self.osc_data = osc_data
        V_in = self.osc_data["r"].flatten()
        x = self.osc_data["frequency"].flatten()
        self.V_out_osc = ceil(1e5*(max(V_in) * np.sqrt(2)))/1e5
        # obtain value of Z_in given the parameters
        Z_in = self.predict(x)
        # Expected impedance of the oscillator
        z = (self.V_out_osc / V_in - 1) * Z_in - self.R_out

        # Do the fitting
        start_time = timeit.default_timer()
        print("Fitting function to {} datapoints ...".format(len(x)))
        # guess = (4e5, 2.2, 596e3*2*np.pi, 1e-1) works well
        popt, _ = curve_fit(fun, x.flatten(), z.flatten(),
                            p0 = guess, maxfev = np.int(1e7),
                            method = "lm")
        end_time = timeit.default_timer()
        print("Found parameters: {}.".format(popt))
        print("Finished in {:.3f} s".format(end_time - start_time))

        self.popt_osc = popt
        z_hat = fun(x, *popt)
        z_hat_corr = analysis.Fitter.apply_correction(self, f = x, r_e = z_hat, popt = popt)
        osc_dict = {   "y_hat": z_hat, "y_hat_corr": z_hat_corr, "y": z, "x": x,
                        "popt": popt}

        # Extract presumed parameters of oscillator
        r = popt[0] # this is not sure, but seems plausible
        I0 = self.V_out_osc / r
        L = (r * popt[1])/popt[2]
        C = 1/((popt[2]**2) * L)
        C0 = popt[3] * C # C0 is parasitic capacitance
        V0 = (1/C) * (I0 / (popt[1]*popt[2]))
        resonator_params = {
            "R": r, "L": L, "C": C, "C0":C0,
            "I0": I0, "V0": V0
        }
        print("Presumed Resonator Parameters: \n{}".format(resonator_params))
        return osc_dict
