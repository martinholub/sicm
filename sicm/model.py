from sicm import io, analysis, plots, filters
from sicm.utils import utils

import numpy as np
import timeit
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

class SICMModel(object):
    """SICM Model

    Model is described by equations (1)-(3) from Chen 2012 [1].

    References
      [1]: https://doi.org/10.1146/annurev-anchem-062011-143203
    """
    def __init__(self, z, U, h, kappa, r_p, r_i, r_o):
        self.R_p = self._calculate_pipette_resistance(h, kappa, r_p, r_i)
        self.R_ac = self._calculate_access_resistance(z, kappa, r_o, r_i)
        self.I = self._calculate_current(U)
        self.z = z
        self.r_i = r_i
        # Assing also other parameters
        self.r_p = r_p
        self.r_o = r_o
        self.h = h
        self.U = U
        self.kappa = kappa


    def _calculate_pipette_resistance(self, h, kappa, r_p, r_i):
        """Calculate pipette resistance

        Pipette resistance is always present and gives bound on current,
        I_max = U/R_p.

        Parameters
        ---------
        h: tip length [m]
        kappa: conductivity of electrolyte [1/(m*Ohm)]
        r_p: inner radius of tip base [m]
        r_i: inner radius of tip openning [m]
        """
        R_p = h / (kappa*r_p*r_i*np.pi)
        return R_p

    def _calculate_access_resistance(self, z, kappa, r_o, r_i):
        """Calculate access resistance due to proximity of tip to surface

        Parameters
        ----------
        z: distance from surface [m]
        kappa: conductivity of electrolyte [1/(m*Ohm)]
        r_o: outer radius of the tip opening [m]
        r_i: inner radius of tip openning [m]
        """
        R_ac = 1.5*np.log(r_o/r_i) / (z*kappa*np.pi)
        return R_ac

    def _calculate_current(self, U):
        """Calculate current of tip-substrate system

        Parameters
        ---------
        U: applied bias [V]

        Returns
        ----------
        I: current [A]
        """
        I = U / (self.R_p + self.R_ac)
        return I

    def plot(self, do_invert = False):
        """Plot analytical relationship

        Parameters
        --------
        do_invert: bool
            Invert x,y axis and plot y vs. x plot
        """
        x = self.z/(2*self.r_i) # scale by diameter
        y = self.I
        y_max = self.U / self.R_p
        # y_max = np.max(y) if y[-1] > 0 else np.min(y)
        leg = "Current-distance relation @ T=298.15K"
        if do_invert:
            plots.plot_generic([y/y_max], [x], [r"$I/I_{max}$"], [r"$z/d$"], leg, "inverted I vs. z curve")
        else:
            plots.plot_generic([x], [y/y_max], [r"$z/d$"], [r"$I/I_{max}$"], leg, "I vs. z curve")

    def fit_wrapper(self):
        """A wraper to define fixed parameters in the scope of fitting function

        Decision on which parameters to treat as fixed and which to fit is largely
        arbitrary. Here the choice is based on ho well the parameters could be related
        between numerical and analytical mode as the description of the analytcial model
        does not necessarily clearly indicated what is meant by what parameter.

        """
        # dont treat these as free variables
        r_i = self.r_i
        r_o = self.r_o
        U = self.U
        # Assume you know what a good guess is, based on analytical modoel
        guess = (self.r_p, self.h, self.kappa)

        def _inverse_fit(i, *params):
            """Fit inverted approach

            Fits function that yields distance for value of current
            """
            r_p, h, kappa = params
            R_p = h / (kappa*r_p*r_i*np.pi)
            # d = f(i)
            ## use for unscaled x, y with guess = [r_p, h, kappa]
            ## or for scaled x=x/(2*r_i), y=y/y_max with guess=[r_p,h*r_i,kappa/np.sqrt((y_max/U))]
            d = (i * 1.5 * np.log(r_i/r_o)*r_p*r_i) / (h*(i-U/R_p))

            # d/(2*r_i) = f(i/R_p)
            ## use this if you fit to scaled values x/(2*r_i), y= y/y_max
            ## with guess = [r_p, h, kappa]
            ## parameters FOR SURE loose physical meaning, thus above approach prefered
            # d = (3*i*R_p*np.log(r_o/r_i)*r_p)/(4*h*(U-R_p*i))
            return d

        return _inverse_fit, guess


    def fit_approach(self, y, x, guess = None):
        """Fits inverted approach

        Parameters
        -----------
        y: array-like
            The values we are trying to fit, dependent data - f(x, ...). Here DISTANCE (z).
        x: array-like
            Independent variable where data is measured. Here CURRENT (i).
        """
        fun, guess_ = self.fit_wrapper()
        if guess: # Supply user defined guess
            guess_ = guess

        #Solve non-linear lsq problem, yielding parama minimizing fun(x,*params)-y
        start_time = timeit.default_timer()
        try:
            print("Fitting {} to {} datapoints ...".format(fun.__qualname__, len(x)))
        except AttributeError as e:
            print("Fitting {} to {} datapoints ...".format(fun.__name__, len(x)))

        popt, _ = curve_fit(fun, x, y, p0 = guess_, maxfev = np.int(1e7),
                            # bounds = ([1e-21, 1e-21, 1e-21], [np.inf, np.inf, np.inf]),
                            method = "lm") # only lm works well

        end_time = timeit.default_timer()
        print("Found parameters: {}.".format(popt))
        print("Finished in {:.3f} s".format(end_time - start_time))
        self.popt = popt

        self.plot_fit(y, x, double_ax = True)

    def predict(self, x, popt = None):
        """Evaluate fitted function at value(s) x
        """
        if popt is None:
            popt = self.popt

        fun, _ = self.fit_wrapper()
        y_hat = fun(x, *popt)
        return y_hat

    def plot_fit(self, y, x, y_hat = None, double_ax = False, plot_range = None):
        """Plots results of function fitting

        Calculates error of the fit as sum of square errors, scaled by maximum
        amplitude in sampled data.

        Parameters
        --------------------
        y: array-like
            Dependent variable
        x: array-like
            Independet variable
        y_hat: array-like
            Predicitons of y,  e.g. f(x, *popt)
        double_ax: bool
            Should the second line be plotted on its own yaxis?
        plot_range: array-like
            Bounds to trim to as [low, high] x-values of.
        """
        if plot_range:
            keep_id = np.logical_and(x > plot_range[0], x <= plot_range[-1])
        else:
            keep_id = np.asarray([True]*len(x))

        if y_hat is None:
            y_hat = self.predict(x, self.popt)
        # Relative squared error
        sqerr = np.sum(np.power((y_hat - y)/np.max(y), 2))
        print("error on fit: {:.9E}".format(sqerr))

        plt.style.use("seaborn")
        fig, ax = plt.subplots(figsize = (6.4, 4.8))

        # Plot amplitude on first axis
        ax.plot(x[keep_id], y[keep_id], label = r"$y$", marker = ".", color = "k",
                alpha = 0.3, markevery = 1, linestyle = "")

        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")
        # Add info box
        txt = r"$error_{{fit}}$: {:.2E}".format(sqerr)
        plt.text(1.1, 0.1, txt, transform=ax.transAxes,
            fontdict = {'color': 'k', 'fontsize': 12, 'ha': 'left', 'va': 'center',
                        'bbox': dict(boxstyle="square", fc="w", ec="k", pad=0.2)})

        # Plot fit on second axis
        ax2 = ax.twinx() if double_ax else ax
        ax2.plot(   x[keep_id], y_hat[keep_id], "r--", label = r"$\hat{y}$ (fit)",
                    alpha = 0.7)

        if double_ax:
            ax2.set_ylabel(r"$\hat{y}$", color="r")
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
