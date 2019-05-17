from sicm import io, analysis, filters
from sicm.plots import plots
from sicm.utils import utils

import numpy as np
import timeit
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


class Model(object):
    def __init__(self):
        pass

    def predict(self, x, popt = None):
        """Evaluate fitted function at value(s) x
        """
        if popt is None:
            popt = self.popt

        fun, _ = self._fit_wrapper()
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

    def _fit(self, guess = None, double_ax = True):
        fun, guess_ = self._fit_wrapper()
        x = self.x
        y = self.y
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

class SICMModel(Model):
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
        super(SICMModel, self).__init__()


    def _calculate_pipette_resistance(self, h, kappa, r_p, r_i):
        """Calculate pipette resistance

        Pipette resistance is always present and gives bound on current,
        I_max = U/R_p. r_p/h is tan(alpha) where alpha is inner opening angle.

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

    def _fit_wrapper(self):
        """A wraper to define fixed parameters in the scope of fitting function

        Decision on which parameters to treat as fixed and which to fit is largely
        arbitrary. Here the choice is based on ho well the parameters could be related
        between numerical and analytical mode as the description of the analytcial model
        does not necessarily clearly indicated what is meant by what parameter.

        """
        # dont treat these as free variables
        r_i = self.r_i

        # Assume you know what a good guess is, based on analytical modoel
        guess = (np.arctan(self.r_p / self.h) * 180/np.pi, self.r_o)

        def _inverse_fit(i, *params):
            """Fit inverted approach

            Fits function that yields distance for value of current. If fitting fails
            try to narrow the data range, this makes it easier to fit a functional
            relationship.
            """
            ## OLD VERSION 1 - Delete at next revision --------------------------
            # guess = (self.r_p, self.h, self.kappa)
            # r_p, h, kappa = params
            # R_p = h / (kappa*r_p*r_i*np.pi)
            ## d = f(i)
            # d = (i * 1.5 * np.log(r_i/r_o)*r_p*r_i) / (h*(i-U/R_p))
            # d = (1.5 * np.log(r_i/r_o)*r_p*r_i) / (h*(1-U/(R_p*i)))

            ## OLD VERSION 2 - Delete at next revision---------------------------
            ## d/(2*r_i) = f(i/R_p)
            ## use this if you fit to scaled values x/(2*r_i), y= y/y_max
            ## with guess = [r_p, h, kappa]
            ## parameters FOR SURE loose physical meaning, thus above approach prefered
            # d = (3*i*R_p*np.log(r_o/r_i)*r_p)/(4*h*(U-R_p*i))

            ## CURRENT VERSION - Parametrs have physical meaning ----------------
            alpha, r_o = params
            tan_alpha = np.tan(alpha * np.pi/180) # convert to radians
            ## unscaled
            # R_p = (1 / (np.pi * kappa * r_i)) * (1 / tan_alpha) # pipette resistance
            # d = (1.5 * np.log(r_i/r_o)*r_i) / (1-U/(R_p*i)) * tan_alpha
            ## scaled d/(2*r_i) and i/i_max
            d = (0.75 * np.log(r_i/r_o)) / (1-(1/i)) * tan_alpha
            ## DEBUG Switched d and i, for debuggign only -----------------------
            # d = (1 + 1.5 * np.log(r_o/r_i) * tan_alpha / i)**(-1) # switch d and i

            return d

        return _inverse_fit, guess

    def fit(self, y, x, guess = None, double_ax = True):
        """Fits inverted approach

        Parameters
        -----------
        y: array-like
            The values we are trying to fit, dependent data - f(x, ...). Here DISTANCE (z).
        x: array-like
            Independent variable where data is measured. Here CURRENT (i).
        """
        fun, guess_ = self._fit_wrapper()
        if guess: # Supply user defined guess
            guess_ = guess

        #Solve non-linear lsq problem, yielding parama minimizing fun(x,*params)-y
        start_time = timeit.default_timer()
        try:
            print("Fitting {} to {} datapoints ...".format(fun.__qualname__, len(x)))
        except AttributeError as e:
            print("Fitting {} to {} datapoints ...".format(fun.__name__, len(x)))

        popt, _ = curve_fit(fun, x, y, p0 = guess_, maxfev = np.int(1e6),
                            # bounds = ([1e-21, 1e-21, 1e-21], [np.inf, np.inf, np.inf]),
                            # bounds = ([0, 0], [np.inf, 1]),
                            method = "lm") # only lm works well

        end_time = timeit.default_timer()
        print("Found parameters: {}.".format(popt))
        print("Finished in {:.3f} s".format(end_time - start_time))
        self.popt = popt

        self.plot_fit(y, x, double_ax = double_ax)

class TemperatureModel(Model):
    """Model of current-distance-temperature dependence"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        super(TemperatureModel, self).__init__()

    @property
    def x(self):
        """Current relative to the bulk value"""
        return self._x
    @x.setter
    def x(self, value):
        # Assure current is being passed as relative.
        if np.max(value) > 1.:
            print("Scaling data to 0..1 range.")
            value = value / np.max(value)
        self._x =  value

    @property
    def y(self):
        """Absolute Temperature"""
        return self._y
    @y.setter
    def y(self, value):
        self._y =  value

    def plot(self, d_sub, r_i, do_invert = False, fname = "I vs. Tsub curve"):
        """Plot data

        Parameters
        --------
        do_invert: bool
            Invert x,y axis and plot y vs. x plot
        """
        y = self.y
        x = self.x
        leg = "Current-Substrate temperature relation @ z/d={:.3f}".format(d_sub / (2*r_i))
        fname = fname + "_{:.3f}".format(d_sub / (2*r_i))
        if do_invert:
            plots.plot_generic([y], [x], [r"$T_{sub}$"], [r"$I/I_{bulk}$"], leg, "inverted_"+ fname)
        else:
            plots.plot_generic([x], [y], [r"$T_{sub}$"], [r"$I/I_{bulk}$"], leg, fname)

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

class Medium(object):
    """Medium in the pipette"""
    def __init__(self, theta, gamma, rho):
        self.theta = theta # young's angle; degree
        self.gamma = gamma # Gas-Liaquid surface tension; N/m
        self.rho = rho # medium relativve density (e.g rho_water - rho_air); kg/m3

class Pipette(object):
    """"Pipette Geometry"""
    def __init__(self, d_body, d_tip,length = None, alpha = None):
        self.length = length # length of pipette; m
        self.d_body = d_body # inener diameter of the capillary; m
        self.d_tip  = d_tip # inner diameter of the tip; m
        self.alpha = alpha # tip opening angle; degree

class CapillaryAction(object):
    """Simplified Model of Capillary Action
    """
    def __init__(self, medium, pipette):
        self.medium = medium
        self.pipette = pipette

    def _calculate_height(self):
        """Capillary suspension, height

        Calculates medium height, h, that will be supported by pipette geometry and
        medium properties.
        """
        # Laplace pressure at lower meniscus, divided by 2kappa
        cos = np.cos(np.deg2rad(self.medium.theta - self.pipette.alpha))
        p_down =  cos / (self.pipette.d_tip / 2)
        # Laplace presusre at upper meniscus, divided by 2kappa
        cos = np.cos(np.deg2rad(self.medium.theta))
        p_up =  cos / (self.pipette.d_body/2)
        # capillary length
        kappa = np.sqrt((self.medium.rho * 9.81) / self.medium.gamma)
        height = 2 / (kappa**2) * (p_down - p_up)

        return height

    def _calculate_alpha(self, height = None):
        """Capillary suspension, alpha

        Calculates pipette opening angle, alpha, that will support column of
        medium in pipette.
        """
        if height is None:
            height = self.pipette.length
            # assume fully filled

        # hydrostatic pressure due to column of height h
        p_hs = height * self.medium.rho * 9.81
        # Laplace presusre at upper meniscus
        cos = np.cos(np.deg2rad(self.medium.theta))
        p_up = (2 * self.medium.gamma * cos / (self.pipette.d_body/2))
        # Cosiunus of theta - alpha
        cos_alpha_hat = (p_hs + p_up) * ((self.pipette.d_tip/2) / (2 * self.medium.gamma))
        # Pipette opening angle that supports height h, in degrees
        alpha = self.medium.theta - np.rad2deg(np.arccos(cos_alpha_hat))
        return alpha

    def _calculate_constriction(self, alpha = None):
        """Capillary suspension, stopping diameter

        Calculates at which diameter will the liquid stop given column height and
        properties of pipette and medium.
        """
        if alpha is None:
            alpha = self.pipette.alpha
        # hydrostatic pressure due to column of height h
        p_hs = self.pipette.length * self.medium.rho * 9.81
        # Laplace presusre at upper meniscus
        cos = np.cos(np.deg2rad(self.medium.theta))
        p_up = (2 * self.medium.gamma * cos / (self.pipette.d_body/2))
        # helper factor
        cos = np.cos(np.deg2rad(self.medium.theta - alpha))
        aux = 2 * self.medium.gamma * cos
        # Radius at which the liquid progression will be arested.
        r_stop = aux / (p_hs + p_up)
        return 2*r_stop

    def plot(self, x, y = None, x_lab = None, y_lab = None, legend = None, fname = None):
        """tba"""
        if y is None:
            y = self._calculate_alpha()
        plots.plot_generic([x], [y], [x_lab], [y_lab], legend, fname)

class Laser(Model):
    """Laser Object for Calibration"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        super(Laser, self).__init__()

    @property
    def fun(self):
        val, _ = self._fit_wrapper()
        return val

    def _fit_wrapper(self):
        """"Fit convenience wrapper"""
        guess = [250, -250]
        def _relu_fit(x, *params):
            """Fit general exponential relationship
            Parameters
            -----------
            x: array-like
                Diode Current [a.u.]
            params: tuple of floats
                Parameters of the model
            """
            c1, c2 = params
            zero = 0
            if len(x) > 1:
                zero = [zero] * len(x)
            y = np.maximum(zero , c1*x + c2)
            return y
        return _relu_fit, guess

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

class GaussianBeam(Model):
    """Fit a gaussian beam model to an image of laser spot
    """

    def __init__(self, impath):
        self.x = io.load_image(impath)
        self.z = z
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
