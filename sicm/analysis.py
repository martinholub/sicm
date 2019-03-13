import numpy as np
import copy
import timeit

import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import curve_fit
from scipy.signal import detrend
from scipy import stats

from sicm.utils import utils
from sicm import plots
from sicm.filters import LowPassButter
from math import ceil

class Picker(object):
    """Object to collect information on clicked points"""
    def __init__(self):
        self.picks = []

    def onpick(self, event):
        """Event invoked on clikcing a point"""
        ind = event.ind[0]
        x, y, z = event.artist._offsets3d
        print("x:{}, y:{}, z:{}".format(x[ind], y[ind], z[ind]))
        self.picks.append({ind: (x[ind], y[ind], z[ind])})

def level_plane(X, Y, Z, is_debug = False, interactive = True, z_lab = "Z(um)"):
    """Level Tilted Plane

    Selection of pints for plane fitting is done interactively to deal with
    less predictable surface topography. The functions must be called from console (not from ipynb) to work properly.

    Parameters
    --------
    X, Y, Z : array-like
        1D arrays of point cooridnates in 3D space

    Returns
    ---------
    X_sq, Y_sq: array-like
        X, Y coordinates of points convertd to 2D matrix
    Z_sq_corr: array-like
        Z coordinates corrected for tilt

    References:
      *http://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
      *https://matplotlib.org/gallery/event_handling/ginput_demo_sgskip.html#sphx-glr-gallery-event-handling-ginput-demo-sgskip-py
      *https://stackoverflow.com/questions/21851114/can-matplotlib-pick-event-return-array-indeces-rather-than-values-or-pixels
      *https://matplotlib.org/api/collections_api.html#matplotlib.collections.Collection

    """
    # TODO: allow non-square matrices
    # Reshape to square matrix
    a = np.int(np.sqrt(len(Z)))
    X_sq = np.reshape(X[:a**2], [a]*2);
    Y_sq = np.reshape(Y[:a**2], [a]*2);
    Z_sq = np.reshape(Z[:a**2], [a]*2);

    ## flip every second column; enables `contourf` plot
    #X_sq[1::2, :] = X_sq[1::2, ::-1]
    #Y_sq[1::2, :] = Y_sq[1::2, ::-1]
    #Z_sq[1::2, :] = Z_sq[1::2, ::-1]

    if interactive:
        # Select points interactively
        with mpl.rc_context(rc={'interactive': True}):
            fig = plt.figure(figsize = (6, 4))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_title("Click on 3 Points To Select Them")
            ax.scatter(X_sq.flatten(), Y_sq.flatten(), Z_sq.flatten(),
                        c = Z_sq.flatten(), marker = "^", picker = 5,
                        alpha = 0.3, cmap = "viridis")
            ax.set_xlabel("X [um]"); ax.set_ylabel("Y [um]"); ax.set_zlabel(z_lab)
            picker = Picker()
            cid = fig.canvas.mpl_connect("pick_event", picker.onpick)
            plt.show()
            input("[Press Enter once you have selected 3 points] \n")
            fig.canvas.mpl_disconnect(cid)
            # vals = plt.ginput(3, show_clicks = True) # just for 2D

        vals = [list(x.values())[0] for x in picker.picks]
        p1 = np.asarray(vals[0])
        p2 = np.asarray(vals[1])
        p3 = np.asarray(vals[2])
    else:
        p1 = np.asarray(((X_sq[1, 1], Y_sq[1, 1], Z_sq[1, 1])))
        p2 = np.asarray((X_sq[1, a-1], Y_sq[1,a-1], Z_sq[1, a-1]))
        p3 = np.asarray((X_sq[a-1, 1], Y_sq[a-1, 1], Z_sq[a-1, 1]))

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    # Compute tilt and correct for it
    Z_tilt = (d - a * X_sq - b * Y_sq) / c
    Z_sq_corr = Z_sq - (Z_tilt - np.min(Z_tilt))

    if is_debug:
        # Visualize selected points, their plane and the correctio
        with mpl.rc_context(rc={'interactive': True}):
            plt.style.use("seaborn")
            fig = plt.figure(figsize = (6, 4))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_title("Points Selected for Tilt Correction")

            ax.scatter(*zip(p1, p2, p3), s = 80, c = "r", marker="^")
            ax.plot_trisurf(X_sq.flatten(), Y_sq.flatten(), Z_sq.flatten(),
                            color = "gray", alpha = 0.2)
            ax.plot_trisurf(X_sq.flatten(), Y_sq.flatten(), Z_tilt.flatten(),
                            color = "red", alpha = 0.2)
            ax.plot_trisurf(X_sq.flatten(), Y_sq.flatten(), Z_sq_corr.flatten(),
                            color = "green", alpha = 0.2)
            ax.set_xticks([], []); ax.set_yticks([], []), ax.set_zticks([], []);
            plt.show()

    return (X_sq, Y_sq , Z_sq_corr)



def correlate_signals(data, ykeys):
    """correlate_signals

    Correlates signals described by ykeys
    """
    assert isinstance(ykeys, (list, tuple, np.ndarray))
    assert len(ykeys) == 2

    y0 = data[ykeys[0]].flatten()
    y1 = data[ykeys[1]].flatten()
    cc = np.corrcoef(y0, y1)
    txt = "Correlation Coefficient ({} vs {}): {:.4f}."
    print(txt.format(ykeys[0], ykeys[1], cc[0, -1]))


def _find_peaks(x, guessid, window_size = 50):
    """Finds indices of peaks in data, using initial guess of their id
    """
    # Figure out if you are looking for minima or maxima
    avg_guess = np.mean(x[guessid])
    avg_all = np.mean(x)
    if avg_guess >= avg_all:
        extrema = "max"
    elif avg_guess < avg_all:
        extrema = "min"

    # Pull out indices of extrma around guessed id and within +- window_size
    # TODO: can you vectorize this?
    idxs = []
    annots = []
    for i in guessid:
        low = np.maximum(i-window_size, 0)
        high = np.minimum(i+window_size, len(x))
        temp = x[low:high]
        if extrema == "max":
            extr_id = np.argmax(temp)
        elif extrema == "min":
            extr_id = np.argmin(temp)
        idxs.append(extr_id + low)
        annots.append(_describe_peak(temp, extr_id))

    return(np.asarray(idxs), annots)

def _describe_peak(seq, idx):
    """Describe peak

    This will produce useless vals for Z, because Z changes peridocically with bigger
    time scale.
    Must pay attention that window_size does not overlap other peak(s).

    Parameters
    -----------
    seq: array-like
        Sequence of values
    idx: int
        Location of peak.

    Returns
    -----------
    baseline: float
        median of all but peak values in sequence
    rel_change: float
        Change between peak and baseline value, scaled by baseline
    """
    baseline = np.median(np.delete(seq, idx))
    rel_change = np.abs((seq[idx] - (baseline)) / baseline)

    return baseline, rel_change

def correct_for_current(data, sel = None, ncorr = 1, window_size = 100):
    """ Try to correct for jumps in current

    Parametrs
    --------------
    data: dict
    sel: array-like
        Indices to values in dict, selecting 'first' or 'last' values.
    ncorr: int
        How many corrections to perform, how many jumps are in the data?
    window_size: int
        Will consider values from [-window_size/2:window_size/+1]

    Return
    """
    window_size = window_size + 10
    data_ = copy.deepcopy(data)
    if ncorr == 0:
        return data_
    else:
        msg = "WARNING: Data will be corrected for jump in current and height.\n" + \
              "The keys corresponding to the values will be appended with '_corrected'."
        print(msg)

    try:
        # Full data, for output
        x_ = data_["Current1(A)"]
        y_ = data_["Z(um)"]
        # Selected data, for math
        if sel is None: sel = np.arange(0, len(x_))
        x = x_[sel]
        y = y_[sel]
    except AttributeError as e:
        raise e # must have these keys

    # Find where the biggest ncorr jumps occur
    grad = np.diff(x.flatten())
    idxs = np.argsort(np.abs(grad))[::-1] # sort decreasing
    idxs = np.sort(idxs[0:ncorr]) # sort increasing
    idxs = np.append(idxs, -1)

    for i in range(len(idxs) - 1):
        # If multiple ncorr, avoid spanning other peaks as well
        prev_id = 0 if i == 0 else idxs[i-1]
        next_id = (len(x) -1) if idxs[i+1] == - 1 else idxs[i+1]
        low_id = np.amax([prev_id, idxs[i] - window_size//2])
        high_id = np.amin([next_id, idxs[i] + window_size//2 + 1])
        # Y, height
        pre_peak = np.average(y[low_id:idxs[i] - 5])
        post_peak = np.average(y[idxs[i] + 5: high_id])
        # X
        x_pre_peak = np.average(x[low_id:idxs[i] - 5])
        x_post_peak = np.average(x[idxs[i] + 5: high_id])
        # correct values
        # Adjust post_peak values by the difference of pre_peak and post peak
        # This is usually fine for current, and mostly acceptable for z if
        # not changing too abruptly.
        x_[sel[idxs[i]] + 1:sel[next_id] + 1] = \
                    x_[sel[idxs[i]] +1:sel[next_id] + 1] + (x_pre_peak - x_post_peak)
        y_[sel[idxs[i]] + 1:sel[next_id] +1 ] = \
                    y_[sel[idxs[i]] +1:sel[next_id] + 1] + (pre_peak - post_peak)

    # reassing
    data_["Current1(A)_corrected"] = x_
    data_["Z(um)_corrected"] = y_
    return data_

class Fitter(object):
    """Class for fitting a Lorentzian curve to frequency sweep data.
    """
    def __init__(self, data, V_out = .1, guess = None, date = None):
        self.data = data
        self.guess = guess
        self.date = date
        self.V_out = V_out

    @property
    def data(self):
        """Data as (f, r, theta) tuple."""
        return self._data
    @data.setter
    def data(self, value):
        assert isinstance(value, (list, tuple)), "Data is not a tuple."
        assert len(value) <= 3, "Data has more than three variables."
        self._data = value

    @property
    def date(self):
        """Date of the experiment"""
        return self._date
    @date.setter
    def date(self, value):
        if value is None:
            value = "00/00/0000 00:00"
        assert isinstance(value, (str, )), "Date is not a string."
        self._date = value

    @property
    def guess(self):
        """Ïnitial guess for curve fit.

        Success of fiting is strongly determined by accuracy of initial guess.
        """
        return self._guess
    @guess.setter
    def guess(self, value):
        if value is not None:
            assert isinstance(value, (list, tuple)), "Guess must be a tuple."
            print("Expecting {} paramters in fit.".format(len(value)))
        self._guess = value


    def lorentzian_fun(f, *params):
        """ Lorenzian function

        Constructs a Lorenzian function of the form y = f(x, *params), following
        equation 1 in [1]. Here, y is the voltage amplitude of electrically driven
        oscillations, measured experimentall and x is frequency in Hz.

        Parameters
        -------------
        f: np.ndarray
            Frequency in Hz. An independent variable for function fit.
        params: tuple, list
            Tuple of (i, Q, w0, CC). Parameters to be fitted. If not supplying
            p0 to curve_fit, must pass all parameters directly (not via *tuple).

        Returns
        -------------
        r_e: np.ndarray
            Amplitude of (electrically-driven) oscilaltions. Dependent variable.

        References
        --------------
          [1]: https://doi.org/10.1063/1.2756125
        """
        w = 2 * np.pi * f # x
        i, Q, w0, CC = params
        # y, also A_e
        r_e = \
            ((i*w)/(Q*w0)) * \
            np.sqrt(
                (1+2*CC*(1-(w/w0)**2)+CC**2*(1-(w/w0)**2)**2+CC**2*(w/(w0*Q))**2) / \
                ((1 - (w/w0)**2)**2+(w/(w0*Q))**2)
            )
        return r_e

    def fit_function(self, fun):
        """Fits function to data

        Fits function fun to data starting from initial guess. Both guess and data
        should have been passed to class constructor. Uses scipy's `curve_fit`.
        `fun` must take same number of additional arguments as is the length of `guess`.

        Whether fitting is successful depends largely on accuracy of initial guess,
        especially for `w0` and `CC` paramaters.

        Returns (Assigns)
        ------------
        popt: tuple
            Values of fitted free parameters in fun, (i0, Q, w0, CC).
        Q: float
            Quality factor. ~200 for QTF in cotact, and ~10k for free QTF.
        f0: float
            Resonance frequency in Hz.

        References
        ----------
          [1]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit
        TODO: decorator logging params of curve_fit call
        TODO: progbar
        """
        x = self.data[0] # frequency
        y = self.data[1] # amplitude
        #Solve non-linear lsq problem, yielding parama minimizing fun(x,*params)-y
        start_time = timeit.default_timer()
        try:
            print("Fitting {} to {} datapoints ...".format(fun.__qualname__, len(x)))
        except AttributeError as e:
            print("Fitting {} to {} datapoints ...".format(fun.__name__, len(x)))

        popt, _ = curve_fit(fun, x, y, p0 = self.guess, maxfev = np.int(1e7))
        # More complicated method, but does not fare  better
        # popt, _ = curve_fit(fun, x, y, p0 = self.guess, method = "trf",
        #     ftol = 1e-12, xtol = 1e-12, maxfev = np.int(1e7), sigma = len(x)*[1e-5],
        #     bounds = ([1e-4, 1e2, 3e4*2*np.pi, 1e-2], [1e1, 2e4, 4e4*2*np.pi, 1e4]))
        end_time = timeit.default_timer()
        print("Found parameters: {}.".format(popt))
        print("Finished in {:.3f} s".format(end_time - start_time))
        self.popt = popt
        self.Q = popt[1]
        self.f0 = popt[2]/(2*np.pi)

        # Extract presumed parameters of oscillator
        v0 = popt[0]
        r = 23750 # https://dx.doi.org/10.3938/NPSM.65.76
        I0 = v0 / r
        L = (r * popt[1])/popt[2]
        C = 1/((popt[2]**2) * L)
        C0 = popt[3] * C # C0 is parasitic capacitance
        V0 = (1/C) * (I0 / (popt[1]*popt[2]))
        resonator_params = {
            "R": r, "L": L, "C": C, "C0":C0,
            "I0": I0, "V0": V0
        }
        print("Presumed Resonator Parameters: \n{}".format(resonator_params))

    def apply_correction(self, theta_e = None, f = np.array([]), r_e = np.array([]),
                        popt = None):
        """Applies correction for stray capacitance

        Implemented as per Lee et al. 2007 [1].

        Parameters
        -----------
        theta_e: array-like
            Phase measured experimentally. Should be in -pi/2 .. pi/2. If None,
            will be calculated from data.
        f: array-like
            Frequency. Optional.
        r_e: array-like
            Amplitude. Optional.

        Returns (Assigns)
        --------------
        r_m: array-like
            Amplitude of mechanically driven oscillations, with stray capacitance
            compensated. Eq 3 in [1].
        theta_m: array-like
            Phase of mechanicaly driven oscillations. Eq 4 in [1].
        theta_e: array-like
            Phase of electrically driven oscillations. Eq 2 in [1].

        References
        ---------
          [1]: https://doi.org/10.1063/1.2756125
        """
        if popt is not None:
            i0, Q, w0, CC = popt
        else:
            i0, Q, w0, CC = self.popt
        if isinstance(f, (float, int, np.float, np.int)):
            pass
        elif f.size == 0 :
            f = self.data[0]
        if isinstance(f, (float, int, np.float, np.int)):
            pass
        elif r_e.size == 0:
            r_e = self.data[1] # could also use fitted values r_e
        w = f * 2 * np.pi
        # Obtain argument(phase) of complex expression
        # note that this is in range -pi/2..pi/2 whereas range of phase in saved data
        # is 0..-pi. Need to adjust one or the other for comparability.
        if theta_e is None:
            theta_e = np.arctan2(
                (1 - (w/w0)**2 + CC*(1-(w/w0)**2)**2 + CC*(w/(w0*Q))**2), w/(w0*Q))
        CV = CC * i0 / (w0 * Q) # CV = C_0*V_0
        # Corrected values
        r_m = np.sqrt(r_e**2 - 2*w*CV*r_e*np.sin(theta_e) + (w*CV)**2)
        theta_m = np.arctan2(
            r_e*np.sin(theta_e) - w*CV, r_e*np.cos(theta_e)
        )
        self.r_m = r_m
        self.theta_m = theta_m
        self.theta_e = theta_e

        return r_m

    def plot_fit(self, fun):
        """Plots results of function fitting

        Calculates error of the fit as sum of square errors, scaled by maximum
        amplitude in sampled data.
        """
        f = self.data[0]
        r = self.data[1]
        r_e = fun(f, *self.popt) # evaluate fun at popt
        # Relative squared error
        sqerr = np.sum(np.power((r_e - r)/np.max(r), 2))
        r_m = self.r_m

        plt.style.use("seaborn")
        fig, ax = plt.subplots(figsize = (6.4, 4.8))
        text = "Lorentzian fit @ {}".format(self.date)
        fig.suptitle(text, size  = 16, y = 0.92)

        # Plot amplitude on first axis
        ax.plot(f, r, label = r"$A_e$ (data)", marker = ".", color = "k", alpha = 0.3,
                markevery = 5, linestyle = "")
        ax.plot(f, r_e, "k--", label = r"$A_e$ (fit)", alpha = 0.7)
        ax.plot(f, r_m, "g-", label = r"$A_m$", alpha = 0.7)
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel("amplitude [V]")
        # Add info box
        txt = "$f_0$: {:.2f} Hz\n$Q$: {:.2f}\n$error_{{fit}}$: {:.2E}".format(\
                        self.f0, self.Q, sqerr)
        plt.text(1.1, 0.1, txt, transform=ax.transAxes,
            fontdict = {'color': 'k', 'fontsize': 12, 'ha': 'left', 'va': 'center',
                        'bbox': dict(boxstyle="square", fc="w", ec="k", pad=0.2)})

        # Plot phase on second axis
        ax2 = ax.twinx()
        ax2.plot(f, self.theta_e, "r--", label = r"$\theta_e$", alpha = 0.3)
        ax2.plot(f, self.theta_m, "r-", label = r"$\theta_m$", alpha = 0.3)
        ax2.set_ylabel("phase [rad]", color="r")
        ax2.tick_params("y", colors="r")
        ax2.grid(axis = "y", color = "r", alpha = .3, linewidth = .5, linestyle = ":")
        # Combine legends and show.
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, bbox_to_anchor = (1.3, 1.1), frameon = True)
        utils.save_fig(text)
        plt.show()

    def process(self):
        """Wrapper for fitting, correcting and plotting freq. sweep data

        Optionally, can pase `theta_e` to `apply_correction` to use meausred values
        instead of computed ones.

        """
        fun = Fitter.lorentzian_fun # must use correct class name
        self.fit_function(fun)
        _ = self.apply_correction() # can pass self.data[2] here
        self.plot_fit(fun)
