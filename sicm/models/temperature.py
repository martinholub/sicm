import os
import numpy as np
import pandas as pd
import timeit
from scipy.optimize import curve_fit
from copy import deepcopy
from itertools import product


from sicm import io
from sicm.plots import plots
from sicm.utils import utils
from .model import Model
from sicm.mathops import mathops

import matplotlib.pyplot as plt

class AnalyticalModel(object):
    """Analytical Model of Ion Current Through Pipette at Elevvated Temperature

    This is a STUB, must be completed later.
    """
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

class TemperatureModelArray(object):
    def __init__(self, adata, tdata, r_subs, Ts, d_pipette, T0, datadir, name, coln = 3):
        self.T0 = T0
        self.d_pipette = d_pipette
        self.r_sub = r_subs
        self.T_sub = Ts
        self.fpath = os.path.join(datadir, name)
        self.model_array = []
        self._build_model_array(adata, tdata, r_subs, Ts, T0, d_pipette, coln)

    def _scale_by_bulk(self, x, y, dT, allow_adjust = True):
        """Scale current measurement by bulk value

        If dT=T-T_bulk is non-zero, fits an exponential to y=f(x) data
        to obtain value of y far from surface.
        """

        if any(dT < -0.1): raise Exception("Decreasing Temperature!")

        if np.min(dT) > 0.1 and allow_adjust:
            # Avoid exponential fit to the elbow region
            ## TODO: How to determine the interval limit robustly???
            sel = x > 2.
            scaler = mathops.find_bulk_val(x[sel], y[sel])
        else:
            scaler = y[-1]
        return y / scaler

    def _build_model_array(self, adata, tdata, r_subs, Ts, T0, d_pipette, coln = 3):
        """Build an iterable of TemperatureModels"""
        model_array = []
        for t in Ts: # Iterate over unique values of T_sub
            xx = [] ; yy = []; TT = []
            for r in r_subs: # Iterate over unique values of rUME
                # Temperature
                dsub2 = tdata[(tdata["Tsub (K)"] == t) & (tdata["rUME (m)"] == r)]
                col_sel = [col for col in dsub2 if col.startswith("Temperature (K)")]
                dT = dsub2[col_sel[0]].values.flatten() - T0
                TT.append(dT)

                # X-Axis
                dsub = adata[(adata["Tsub (K)"] == t) & (adata["rUME (m)"] == r)]
                x = dsub["d (m)"].values / d_pipette
                xx.append(x)
                # Y-AXIS
                y = np.abs(dsub.iloc[:, coln].values)
                y = self._scale_by_bulk(x, y, dT, allow_adjust = True)
                yy.append( y)

            # Asign instance of Temperature model
            tm = TemperatureModel(xx, yy, TT, T0, t, r_subs/d_pipette, d_pipette)
            self.model_array.append(tm)

    def plot_overview(self):
        """Plot all approaches for given rUME,T_sub"""
        for m in self.model_array:
            suffix = "_T{:.2f}K_d{:.0f}nm".format(m.T_sub, self.d_pipette*1e9)
            fname = utils.make_fname(self.fpath, suffix.replace(".", "p"))
            m.plot(fname = fname)

    def plot_check(self, plot_approach = False, **kwargs):
        """Visually inspect validity of fits."""
        try:
            for i, m in enumerate(self.model_array):
                suffix = "_T{:.2f}K_d{:.0f}nm_Fit".format(m.T_sub, self.d_pipette*1e9)
                fname = utils.make_fname(self.fpath, suffix.replace(".", "p"))
                m.check_fit(fname, plot_approach, **kwargs)
        except Exception as e:
            print("Model must be fitted before check-plotting!")
            raise e

    def fit(self, err_lim = 2e-1):
        """Fit a linear relationship to select part of data

        Parameters
        -----------
        err_lim: float
            Maximum LSQ error for linear fit.
        """
        if not isinstance(err_lim, (list, tuple, np.ndarray)): err_lim = [err_lim]
        if len(err_lim) != len(self.model_array): err_lim = err_lim * len(self.model_array)
        self.popts = {}
        self.lims = {}
        for i, m in enumerate(self.model_array):
            m.fit(err_lim = err_lim[i])
            lims = np.asarray([x[i] for i, x in zip(m.idx, m.x)])
            self.popts["{:.2f}".format(m.T_sub)] = m.popt
            self.lims["{:.2f}".format(m.T_sub)] = lims

    def extract_valid_fit(self, do_plot = False):
        """Extract mean parameters of the fit from multiple simulations"""

        if do_plot:
            fig, ax = plt.subplots(1, 1, figsize = (4, 4))
            # cs = ["c", "g", "whitesmoke", "y", "b", "pink", "m"]
            # cs += ["brown", "teal", "skyblue", "k", "darkgray", "indigo"]
            # fmts = ["-" + c for c in cs]
            fmts = ["sk"]

        x_ax = np.linspace(1, 1.3, 30)
        a = []; b = []
        info_dict = {}
        summary = {"good": [], "bad": []}
        lims_good = []
        Ys = []
        fun, _ = self.model_array[0]._fit_wrapper("predict")
        for j, (k, super_v) in enumerate(self.popts.items()): # loop over temperatures
            xs = []; ys = []
            info_dict[k] = {"bad": [], "good": []}
            for i, p in enumerate(super_v): # loop over r_sub
                # compute values (dT) of the fit
                y = fun(x_ax, *p) #     y = p[0] * x_ax + p[1]

                # Report validity of the fit
                v1 = self.r_sub[i] / self.d_pipette
                v2 = self.lims[k][i]
                info = {"rSUB/dTIP": "{:.2f}".format(v1), "z/d LIM": "{:.2f}".format(v2)}
                ## Mark data where we do not start @ 0 as bad.
                if False: #y[0] > 0.25 or y[0] < -0.05:
                    info_dict[k]["bad"].append(info)
                    summary["bad"].append((k, "{:.2f}".format(v1), "{:.2f}".format(v2)))
                    continue
                else:
                    info_dict[k]["good"].append(info)
                    summary["good"].append((k, "{:.2f}".format(v1), "{:.2f}".format(v2)))
                    lims_good.append(v2)

                a.append(p[0]); b.append(p[1])
                xs.append(x_ax); ys.append(y); Ys.append(y)

            ## OLD VERSION
            # if do_plot:
            #     plots.plot_generic(xs, ys, ["$I / I_{bulk}$"], ["$\Delta\ T$"],
            #                     fmts = fmts, linewidth = 0.5, alpha = 0.15, ax = ax)
        if do_plot:
            y_temp = np.asarray(Ys).T
            y_mean = np.mean(y_temp, axis = 1)
            y_std = np.std(y_temp, axis = 1)
            plots.errorplot_generic([x_ax], [y_mean], [y_std], [r"$I / I_{bulk}$"],
                                    ["$\Delta\ T$"], fmts = fmts, alpha = 0.4,
                                    ax = ax, markersize = 1)

        import pdb; pdb.set_trace()
        a_, b_ = mathops.get_fit_params(a, b, force_positive = False)
        txt = "y = x*{:.4f} + {:.4f}".format(a_, b_)
        txt += "; stds: ({:.4f}, {:.4f})".format(np.std(a), np.std(b))
        txt += "\nz/d LIM = {:.4f}; (std = {:.4f})".format(np.mean(lims_good), np.std(lims_good))
        print(txt)
        info_dict["summary"] = summary
        info_dict["fit"] = {"coeff (mean)": (a_, b_),
                            "coeff (std)": (np.std(a), np.std(b))}
        if do_plot:
            # Compute valid fit
            # y = np.mean(a) * x_ax + np.mean(b)
            y = fun(x_ax, a_, b_)
            ax.plot(x_ax, y, alpha = 0.75, color = "red", linewidth = 2)
            txt = r"$n = {:d}$".format(len(summary["good"]))
            ax.text(0.75, 0.1, txt, transform = ax.transAxes, color = "black")
        # Save dictionary
        fname = utils.make_fname(self.fpath, "_Annotation", ".json")
        utils.save_dict(info_dict, fname)

class TemperatureModel(Model):
    """Model of current-distance-temperature dependence"""
    def __init__(self, x, y, T, T0 = 298.15, T_sub = None, r_sub = None, d_pipette = None):
        self.T0 = T0 # Ambient T in K
        self.r_sub = r_sub # substrate size, relative to d_pipette
        self.T_sub = T_sub
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
                value=[ self._model_temperature(v,x,r) for v,x,r in \
                        zip(value, self.x, self.r_sub)]
                assert np.max(value) < 100, msg
            elif isinstance(value[0], (list, tuple, np.ndarray)):
                assert all([np.max(v) < 100 for v in value]), msg
        else:
            if isinstance(value, (np.int, np.float)):
                value = [self._model_temperature(value, self.x)]
        self._T =  value

    @property
    def T_sub(self):
        """Temperature of the substrate"""
        return self._T_sub
    @T_sub.setter
    def T_sub(self, value):
        if value is None:
            # Guess temperature of substrate from data
            value = self.T0 + np.max(self.T)
        self._T_sub = value

    def _model_temperature(self, T_sub, x, r_sub = None):
        """Apply model of temperature"""
        T0 = self.T0
        try:
            x += r_sub
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
            ## Can play tricks with scaling but not needed
            # K = self.T0 - 140.0
            # alpha = np.log10(y)
            # y_ = K * (alpha * K / T + alpha)
            ## Plot dT vs I/I_bulk
            xx.append(y)
            yy.append(T)

            leg = "$r_{{sub}}={:.2f} \cdot d_{{tip}}$".format(r)
            legs.append(leg)
        txt = "$T_{{sub}}={:.2f} K$" + "\n" + "$d_{{tip}}={:.0f}nm$"
        txt = txt.format(np.max(self.T) + self.T0, self.d_pipette * 1e9)
        # [r"$z/d$"], [r"$I \cdot (I_{bulk}\ \Delta T)^{-1}$"]
        plots.plot_generic( xx, yy, ["$I / I_{bulk}$"], ["$\Delta\ T$"],
                            legs, fname, scale = "", ticks = True,
                            text = txt, text_loc = (1.01, 0.9), fmts = fmts_buff,
                            invert = None)

    def _fit_wrapper(self, what = "fit"):
        """"Fit convenience wrapper"""
        guess = [1, -1] # guess of likely parameters
        def _linear_fit(x, *params):
            """Fit linear realtionship"""
            a, b = params
            y = a*(x) + b
            return y
        def _linear_predict(x, *params):
            a, b = params
            y = a*(x) + b
            y[y<0] = 0
            return y

        if what == "fit":
            return _linear_fit, guess
        else:
            return _linear_predict, guess

    def _find_elbow(self, x, y, is_debug = False):
        """Depreceated, does not work optimally for required goal

        Delete on next revision
        """
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
        return idx, max_dist

    def _shrink_line_iteratively(self, x, y, error = 5e-2, is_debug = False):
        """Shrink line iteratively from its start"""
        max_dist = np.inf
        x_ = deepcopy(x)
        y_ = deepcopy(y)
        weights_ = np.arange(len(x_), 0, -1) / len(x_)
        for i in range(np.int(1e3)):
            # _, max_dist = self._find_elbow(x_, y_, is_debug)
            weights = np.ones_like(x_)

            popt, fit_err, _, _, _ = np.polyfit(x_, y_, deg = 1, w = weights,
                                                full = True, cov = False)

            # TPunish error on proximal points more severly
            w = weights_[:len(x_)]
            fit_err = np.sum((np.power(w * (popt[0]*x_ + popt[1] - y_), 2)))
            # fit_err = np.sum(np.sqrt(np.diag(cov)))

            if fit_err < error: break
            x_ = x_[1:]
            y_ = y_[1:]
        if is_debug:
            print("Error on fit = {}, popt = ({}).".format(fit_err, popt))
        idx = len(x) - len(x_) - 1
        return x_, y_ , idx

    def _trim_to_quasilinear(self, x, y, error = 5e-2, is_debug = False):
        """Trim data such that they obey quasilinear relationship

        DEPRECEATED!

        Parameters
        --------------
        x: array-like
            Usually relative current, independent variable
        y: array-like
            Usually dT, dependent variable
        """
        # Iteratively remove data from close-to-surface end, until you have somewhat straight line
        # TODO: Check from which end are you removign the data!!!
        max_dist = 1
        idx = 0
        for i in range(100):
            idx_, max_dist = self._find_elbow(x, y, is_debug)
            if max_dist < error: break
            x = x[idx_:]; y = y[idx_:]
            idx += idx_
        return x, y, idx

    def _report_domain_of_validity(self, x, y, z, idx, is_debug = False):
        """Report what is the smallest value of z/d until where the fit is reliable"""

        thresh = z[idx]
        msg = "Relative (z/d) Validity Threshold = {:.2f}".format(thresh)
        print(msg)

    def _select_data(self, idx = 0, err_lim = 5e-2):
        """Select Data for Fitting

        Parameters
        --------------
        idx: int or array-like

        Returns
        --------------
        x,y: array-like
            independent, dependent variables
        """
        if idx is not None:
            # Pull out index and Impose "from->to surface" order
            x = self.y[idx]# [::-1] # relative current is input
            y = self.T[idx]# [::-1] # temperature is output
            z = self.x[idx]# [::-1] # z is relative distance from surface
            xx = deepcopy(x); yy = deepcopy(y)
            x, y, thresh_idx = self._shrink_line_iteratively(x, y, err_lim)
            # self._report_domain_of_validity(xx, yy, z, thresh_idx)
            self.idx = [thresh_idx]
            return [x], [y], [xx], [yy]
        else:
            x = []; y = []; xx = []; yy = []; idxs = []
            for x_, y_, z_ in zip(self.y, self.T, self.x):
                #x_ = x_[::-1]; y_ = y_[::-1]; z_ = z_[::-1]
                xx.append(deepcopy(x)); yy.append(deepcopy(y))
                x_, y_, thresh_idx = self._shrink_line_iteratively(x_, y_, err_lim)
                # self._report_domain_of_validity(xx[-1], yy[-1], z_, thresh_idx)
                x.append(x_); y.append(y_)
                idxs.append(thresh_idx)
            self.idx = idxs
            return x, y, xx, yy

    def fit(self, guess = None, idx = None, double_ax = False, do_plot = False,
            err_lim = 5e-2, verbose = False):
        fun, guess_ = self._fit_wrapper()
        x_, y_, xx_, yy_ = self._select_data(idx, err_lim)
        if guess is None:
            guess = guess_
        #Solve non-linear lsq problem, yielding parama minimizing fun(x,*params)-y
        start_time = timeit.default_timer()
        popts = []
        for x, y, xx, yy in zip(x_, y_, xx_, yy_):

            if verbose:
                try:
                    print(  "Fitting {} to {} datapoints ...".\
                            format(fun.__qualname__.split(".")[-1], len(x)))
                except AttributeError as e:
                    print("Fitting {} to {} datapoints ...".format(fun.__name__, len(x)))

            bounds = (-np.inf, np.inf) #(0, np.inf)
            popt, _ = curve_fit(fun, x, y, p0 = guess, maxfev = np.int(1e6),
                                bounds = bounds, method = "lm")
            # polyfit works equaly well for simple problems
            # popt = np.polyfit(x, y, 1)

            end_time = timeit.default_timer()
            if verbose:
                print("Found parameters: {}.".format(popt))
                print("Finished in {:.3f} s".format(end_time - start_time))
            if do_plot:
                self.popt = popt # need to assing here to make plot_fit work as elswhere
                self.plot_fit(yy, xx, double_ax = double_ax)
            popts.append(popt)
        self.popt = popts # assign the proper fitted values

    def check_fit(self, fname = None, plot_approach = False, **kwargs):
        """Validate your approach"""
        nrows = np.int(np.ceil(len(self.popt) / 2))
        ncols = 2 if (len(self.popt) > 1) else 1
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
                                figsize = (ncols * 5, nrows * 4))
        try:
            axs = axs.flatten()
        except AttributeError as e:
            axs = [axs]
        for i, (x,y,z,p,idx) in \
            enumerate(zip(self.y, self.T, self.x, self.popt, self.idx)):
            # idx = len(x) - (idx + 1)
            xx = []
            yy = []
            zz = []
            legs = []
            fmts = []

            # Data
            xx.append(x[idx:])
            yy.append(y[idx:])
            zz.append(z[idx:])
            legs += ["data (valid)"]
            fmts += [":green"]

            xx.append(x[0:idx+1])
            yy.append(y[0:idx+1])
            zz.append(z[0:idx+1])
            legs += ["data (invalid)"]
            fmts += [":y"]

            # Fit
            if not plot_approach:
                xx.append(x[idx:])
                yy.append(x[idx:] * p[0] + p[1])
                zz.append(np.ones_like(x[idx:]))
                legs.append("fit")
                fmts.append("-whitesmoke")

            # Limiting point
            xx.append(np.asarray(x[idx]))
            yy.append(np.asarray(y[idx]))
            zz.append(np.asarray(z[idx]))
            legs += ["valid. limit"]
            fmts += ["^red"]

            x_lab = ["$I / I_{bulk}$"]
            y_lab = ["$\Delta\ T$"] if not plot_approach else [r"$z/d$"]
            txt = r"$r_{{sub}}={:.2f} \cdot d_{{tip}}$"
            txt += "\n" + r"$\frac{{z}}{{d}}|_{{lim}} = {:.2f}$"
            txt = txt.format(self.r_sub[i], z[idx])
            txt += r", $I_{{rel}}^{{lim}} = {:.2f}$".format(x[idx])
            if not plot_approach:
                sign = "+" if p[1] > 0 else ""
                txt += "\n" + r"$y = {:.2f} x {} {:.2f}$".format(p[0], sign, p[1])

            Y = yy if not plot_approach else zz
            scale = None if not plot_approach else "logy"
            invert = None if not plot_approach else "y"

            plots.plot_generic(xx, Y, x_lab, y_lab, legend = legs, ax = axs[i],
                                fmts = fmts, text = txt,
                                scale = scale, invert = invert, **kwargs) #(0.1, 0.90)

        title = "$T_{{sub}} = {:.2f} K$".format(self.T_sub) + "\n" + \
                "$d_{{pipette}} = {:.0f} nm$".format(self.d_pipette * 1e9)

        fig.suptitle(title, size = 10, y = 1.0 - nrows*0.02)

        suffix = "" if not plot_approach else "_app"
        if fname is not None:
            utils.save_fig(fname, suffix, ".png")
