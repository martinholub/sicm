import os
import numpy as np
import pandas as pd
import timeit
from scipy.optimize import curve_fit
from copy import deepcopy


from sicm import io
from sicm.plots import plots
from sicm.utils import utils
from .model import Model

import matplotlib.pyplot as plt

class AnalyticalModel(object):
    """Analytical Model of Ion Current Through Pipette at Elevvated Temperature"""
    def __init__(self, c0 = 0.25, T = 298.15):
        self.c0 = c0 # Molar
        self.T0 = T0 # K

    @property
    def m_values(self):
        x = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        return np.asarray(x)

    @property
    def y_KCl_values(self):
        x = [   0.965,0.951,0.927,0.901,0.869,0.816,0.768,0.717,0.687,\
                0.665,0.649,0.636,0.626,0.617]
        return np.asarray(x)

    @property
    def M_salt(self):
        return 74.55e-3 # kg/mol

    @property
    def DCl_inf(self):
        return 2.032e-9

    @property
    def DK_inf(self):
        return 1.957e-9

    @property
    def d(self):
        """What is this parameter???"""
        return 1 # kg/L

    @property
    def c0_values(self):
        x = (self.d * self.m_values) / (1 + self.m_values * self.M_salt)
        return x

    def viscosity(self, value):
        nu = 2.414e-5 * (10 ** (247.8 / (value - 140.)))
        return nu

    def intfun_KCl(self, values, is_deriv = False):
        """
        From Hammer & Wu
        TODO: Which paper is that???
        f(c0)
        """
        x = self.c0_values
        y = np.log10(self.y_KCl_values)
        if is_deriv:
            return np.interp(values, x[1:], np.diff(y))
        else:
            return np.interp(values, x, y)
        return

    def DK_c(self, value):
        # value = self.c0
        grad = np.self.intfun_KCl(value, is_deriv = True)
        return self.DK_inf * (1 + value * grad)

    def DCl_c(self, value):
        # value = self.c0
        grad = np.self.intfun_KCl(value, is_deriv = True)
        return self.DCl_inf * (1 + value * grad)

    def DK(self, T):
        """Diffusivity of K at c0 as function of temperature"""
        val = self.DK_c(self.c0)
        val *= (T * self.viscosity(self.T0)) /(self.T0 * self.viscosity(T))
        return val

    def DCl(self, T):
        """Diffusivity of Cl at c0 as function of temperature"""
        val = self.DCl_c(self.c0)
        val *= (T * self.viscosity(self.T0)) /(self.T0 * self.viscosity(T))
        return val


class TemperatureBulkModel(object):
    def __init__(self):
        pass

class HotSpotModel(object):
    def __init__(self):
        pass

class TemperatureModel(Model):
    """Model of current-distance-temperature dependence"""
    def __init__(self, x, y, T, T0 = 298.15, r_sub = None, d_pipette = None):
        self.T0 = T0 # Ambient T in K
        self.r_sub = r_sub # substrate size, relative to d_pipette
        self.d_pipette = d_pipette # pipette diam in m
        self.y = y # relative current
        self.x = x # relative distance from surface
        self.T = T # \Delta T above ambient
        super(TemperatureModel, self).__init__()

    @property
    def y(self):
        """Current relative to the bulk value"""
        return self._y
    @y.setter
    def y(self, value):
        msg = "Current must be passed in as relative."
        if isinstance(value, (list, tuple)):
            assert all([np.max(i) >=1 for i in value]), msg
        else:
            assert np.max(value) >= 1, msg
            value = [value]
        self._y =  value

    @property
    def x(self):
        """Separation From Surface"""
        return self._x
    @x.setter
    def x(self, value):
        msg = "Distance must be passed as relative."
        if isinstance(value, (list, tuple)):
            assert all([np.max(i) >=1 for i in value]), msg
        else:
            assert np.max(value) >= 1, msg
            value = [value]
        self._x =  value

    @property
    def r_sub(self):
        """Radius of hemispherical substrate."""
        return self._r_sub
    @r_sub.setter
    def r_sub(self, value):
        msg = "Size of substrate must be given as relative to pipette diameter."
        if isinstance(value, (list, tuple, np.ndarray)):
            assert all([v > 1e-3 for v in value]), msg
        else:
            assert value > 1e-3, msg
            value = [value]
        self._r_sub = value

    @property
    def T(self):
        """Temperature"""
        return self._T
    @T.setter
    def T(self, value):
        msg = "Temperature must be passed as difference w.r.t bulk."
        if isinstance(value, (list, tuple)):
            if isinstance(value[0], (np.int, np.float)):
                if len[value[0]] == 1:
                    value = [self._model_temperature(v) for v in value]
                else:
                    assert np.max(value) < 100, msg
                    value = [value]
            elif isinstance(value[0], (list, tuple, np.ndarray)):
                assert all([np.max(v) < 100 for v in value]), msg
        else:
            if isinstance(value, (np.int, np.float)):
                value = [self._model_temperature(value)]
        self._T =  value

    def _model_temperature(self, T_sub):
        """Apply model of temperature"""
        T0 = self.T0
        x = self.x
        try:
            # First coordinate where we get temperature
            # r_sub = np.asarray(self.r_sub) + np.min(x)
            # TODO: The above approach is probably wrong!
            raise NotImplementedError()
        except Exception as e:
            r_sub = np.min(x)

        A = (T_sub - T0) * r_sub
        T = A / x
        return T

    def plot(self, fname = None):
        """Plot data

        Parameters
        --------
        fname: str, None
        """
        xx = []
        yy = []
        legs = []
        fmts_buff = [":y", ":g", ":r", ":b", ":m", ":c", ":gray", ":k"]
        for x, y, T, r in zip(self.x, self.y, self.T, self.r_sub):
            xx.append(y)
            # K = self.T0 - 140.0
            # alpha = np.log10(y)
            # y_ = K * (alpha * K / T + alpha)
            yy.append(T)
            leg = "$r_{{sub}}={:.2f} \cdot d_{{tip}}$".format(r)
            legs.append(leg)
        txt = "$T_{{sub}}={:.2f} K$" + "\n" + "$d_{{tip}}={:.0f}nm$"
        txt = txt.format(np.max(self.T) + self.T0, self.d_pipette * 1e9)
        # [r"$z/d$"], [r"$I \cdot (I_{bulk}\ \Delta T)^{-1}$"]
        plots.plot_generic( xx, yy, ["$I / I_{bulk}$"], ["$\Delta\ T$"],
                            legs, fname, scale = "", ticks = True,
                            text = txt, text_loc = (0.1, 0.5), fmts = fmts_buff,
                            invert = None)

    def _fit_wrapper(self):
        """"Fit convenience wrapper"""
        guess = -1, 3
        def _linear_fit(x, *params):
            """Fit linear realtionship"""
            a, b = params
            y = a*x + b
            return y
        return _linear_fit, guess


    def _find_elbow(self, x, y, is_debug = True):
        # https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1
                    # 2D coordinate
        all_coords = np.vstack((x, y)).T
        # Vector between endpoints of line
        line_vec = all_coords[-1] - all_coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        # Distance from first point
        vec_from_first = all_coords - all_coords[0]
        scalar_product =   np.sum(vec_from_first * \
            	           np.matlib.repmat(line_vec_norm, len(y), 1), axis = 1)

        vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        # distance to line is the norm of vec_to_line
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        # knee/elbow is the point with max distance value
        idx = np.argmax(dist_to_line)

        max_dist = dist_to_line[idx]

        if is_debug:
            fig, ax = plt.subplots(1, 1, figsize = (4, 4))
            ax.plot(x, y, label='Data', color='k')
            ax.plot(x, dist_to_line, label='Distances', color='gray')
            ax.plot(x[idx], y[idx], "sr", label='Knee')
            ax.legend()
            plt.show()

        # trim to straight line at return
        return x[:idx], y[:idx], max_dist, idx


    def _trim_to_quasilinear(self, x, y, max_dist_lim = 5e-2, is_debug = False):
        """Trim data such that they obey quasilinear relationship

        TODO: Make max_dist_lim soemwhat depednent on data??
        TODO: Check from which side you are trimming!!!

        Parameters:
        x: array-like
            Usually relative current, independent variable
        y: array-like
            Usually dT, dependent variable
        """
        max_dist = 1
        # Iteratively remove data from close-to-surface end, until you have somewhat straight line
        # TODO: Check from which end are you removign the data!!!
        while max_dist > max_dist_lim:
            x, y, max_dist, idx = self._find_elbow(x, y, is_debug)
        return x, y, idx

    def _report_domain_of_validity(self, x, y, z, idx, is_debug = False):
        """Report what is the smallest value of z/d until where the fit is reliable"""

        thresh = z[idx]
        msg = "Relative (z/d) Validity Threshold = {:.2f}".format(thresh)
        print(msg)

        if is_debug:
            # TODO: Play around with this plot to make sure it is correct!
            plots._set_rcparams(ticks = True)
            fig, axs = plt.subplots(1, 2, figsize = (8, 4))
            axs = axs.flatten()
            axs[0].plot(x, y, "-k")
            axs[0].plot(x[idx], y[idx], "sr")

            axs[1].plot(z, x/y, "-k")
            axs[1].plot(z[idx], (x/y)[idx], "sr")
            axs[1].set_xscale("log");


    def _select_data(self, idx = 0):
        """Select Data for Fitting

        Parameters
        --------------
        idx: int or array-like

        Returns
        --------------
        x,y: array-like
            independet, dependent variables
        """
        if idx is not None:
            # TODO: Check if inversion orders the data properly!
            # Invert to go have dece=reasing separation from surface
            x = self.y[idx][::-1] # relative current is input
            y = self.T[idx][::-1] # temperature is output
            z = self.x[idx][::-1] # z is relative distance from surface
        else:
            x = self.y[::-1]; y = self.T[::-1]; z = self.x[::-1]

        xx = deepcopy(x); yy = deepcopy(y)
        x, y, thresh_idx = self._trim_to_quasilinear(x, y)
        self._report_domain_of_validity(xx, yy, z, thresh_idx, True)
        return x, y, xx, yy

    def fit(self, guess = None, idx = None, double_ax = False):
        fun, guess_ = self._fit_wrapper()
        x, y, xx, yy = self._select_data(idx)
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
        # polyfit works equaly well for simple problems
        # popt = np.polyfit(x, y, 1)

        end_time = timeit.default_timer()
        print("Found parameters: {}.".format(popt))
        print("Finished in {:.3f} s".format(end_time - start_time))
        self.popt = popt

        self.plot_fit(yy, xx, double_ax = double_ax)
