import os
import numpy as np
import pandas as pd
import timeit
from scipy.optimize import curve_fit


from sicm import io
from sicm.plots import plots
from sicm.utils import utils
from .model import Model
from sicm.sicm import Approach, ApproachList


import matplotlib.pyplot as plt

class ParametricSweepGenerator(object):
    """Generate File to be Loaded As Parametric Sweep Parameters"""
    def __init__(self, param_dict):
        self.param_dict = param_dict

    def parse_parameters(self):
        """
        Parameters
        --------------
        param_dict: dictionary
            {name: values} pairs

        References
        ---------------
        [1]  https://www.comsol.ch/forum/thread/attachment/340052/example-df96ce6.txt
        """
        # rUME rUME*range(0.9,0.2,2.1) [m]
        # d "(rt*(10^(range(-2, 4.3/25,2.3)))+rUME)" [m]
        # Tsub range(T0+3.5,0.5,T0+5) [K]
        pass

class ComsolData(object):
    """Data Exported from COMSOL"""
    def __init__(self, datadir, exp_name):
        self.name = exp_name
        self.datadir = datadir
        self.data, self.date = self._load_data(datadir, exp_name)

    def _load_data(self, datadir, exp_name):
        """Load results of a comsol parametric sweep"""
        fpath = os.path.join(datadir, exp_name)
        data, date = io.load_comsol(fpath)
        return data, date


class ComsolModel(Model):
    """Comsol Model of Thermometry

    Parameters
    ---------------
    datadir: str
    exp_name: str
    """
    def __init__(self, datadir, exp_name):
        self.comsol = ComsolData(datadir, exp_name)
        self.approach = None
        super(ComsolModel, self).__init__()

    def add_approach(self, datadir, exp_name, z_move = 25, preserve_overlap = True):
        """Add approach data

        In order to test validity of numerical model, add experimentally obtained
        approach curve.

        Parameters
        ------------
        datadir: str or list
        exp_name: str or list
            If list, multiple approaches are stitched together.
        z_move: int
            Distance (in um) between consecutive approaches (if multiple)
        preserve_overlap: bool
        """
        if isinstance(datadir, (list, )) or isinstance(exp_name, (list, )):
            app_list  = ApproachList(datadir, exp_name)
            self.approach = app_list.stitch(z_move = z_move,
                                            preserve_overlap = preserve_overlap)
            # Check that stitching works consistently
            assert len(set([v.shape[0] for v in self.approach.dsdata.values()])) == 1
        else:
            self.approach = Approach(datadir, exp_name)

    def _select_variable(self, variable = None):
        """Select variable to plot grid for """
        if variable is not None:
            return variable, None
        else:

            keys = list(self.comsol.data.keys())
            n_repeats = [len(self.comsol.data[k].unique()) for k in keys]
            valid_keys  = [k for k, nr in zip(keys, n_repeats) if nr > 0 and k != "d (m)"]
            validator = ['rUME (m)', 'Tsub (K)', 'aspect', 'P_beam (W/m^2)']
            valid_keys = [k for k in valid_keys if k in validator]
            if len(valid_keys) == 1:
                return valid_keys[0], None
            else:
                return valid_keys[1], valid_keys[0]

    def plot_grid(self, pipette_diameter = 220, show_experiment = True, variable = None,
                    col = 2, offset = 0, add_unity_line = False, window_size = 0, **kwargs):
        """Plot all numerical results on grid

        Displays result of numerically simulated approach at different temperatures.
        If experimental data avialable it is superimposed in all plots.
        """
        var_name, secondary_var_name = self._select_variable(variable)
        try:
            uniqs = self.comsol.data[var_name].unique()
        except KeyError as e:
            uniqs = [None]
        nrows = np.int(np.ceil(len(uniqs) / 2))
        ncols = 2 if len(uniqs) > 1 else 1
        fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
                                figsize = (ncols * 5, nrows * 4))
        try:
            axs = axs.flatten()
        except AttributeError as e:
            axs = [axs]

        title = "Comsol Param Grid" + "\n" + \
                "Model: {} @ {}".format(self.comsol.name, self.comsol.date)

        for i, var in enumerate(uniqs):
            # Look at given temperature
            if var is None:
                sel = [True] * len(self.comsol.data)
            else:
                sel = self.comsol.data[var_name] == var

            # TODO: Simplify treatmement of secondary variable, combine with simple
            if secondary_var_name is not None:
                # Attempt to visualize experiments with multiple variables
                x = []
                y = []
                fmts_buff = [":y", ":g", ":r", ":b", ":m", ":c", ":gray", ":k"]
                fmts = []
                legend = []
                uniqs_second = self.comsol.data[secondary_var_name].unique()
                # Loop over subsets for whic the secondary variable is unique
                for j, sec_var in enumerate(uniqs_second):
                    sel_ = np.logical_and(self.comsol.data[secondary_var_name] == sec_var, sel)
                    # De everything as if you are dealing with simple dataset
                    y_sec = np.abs(self.comsol.data[sel_].iloc[:, col].values.flatten())
                    y_sec = y_sec / y_sec[-1]
                    sel2 = y_sec >= 0.97
                    y_sec = [y_sec[sel2]]
                    x_sec = self.comsol.data["d (m)"][sel_].values.flatten()[sel2]
                    if offset is None:
                        offset = self.comsol.data["d (m)"][sel].unique().min()
                    x_sec -= offset
                    x_sec = [x_sec / (pipette_diameter * 1e-9)]
                    # Grow the list of items for plotting
                    x += x_sec
                    y += y_sec
                    fmts += [fmts_buff[j]]

                    if secondary_var_name.lower().startswith("t"):
                        txt = "T = {:.2f} K".format(sec_var)
                        if var > 1e-6:
                            ax_txt = r"$r_{{sub}}$={:.2f}$\mu m$".format(var * 1e6)
                        else:
                            ax_txt = r"$r_{{sub}}$={:.0f}$nm$".format(var * 1e9)

                    elif secondary_var_name.lower().startswith(("r")):
                        if sec_var > 1e-6:
                            txt = r"$r_{{sub}}$={:.2f}$\mu m$".format(sec_var * 1e6)
                        else:
                            txt = r"$r_{{sub}}$={:.0f}$nm$".format(sec_var * 1e9)

                        if var_name.lower().startswith("t"):
                            ax_txt = "T = {:.2f} K".format(var)
                        elif var_name.lower().startswith("r"):
                            ax_txt = r"$r_{{sub}}$={:.2f}$\mu m$".format(var * 1e6)
                        elif var_name.lower().startswith("p"):
                            ax_txt = "P = {:.2f} $W/cm^2$".format(var * 1e-4)
                        else:
                            ax_txt = None

                    elif secondary_var_name.lower().startswith(("a")):
                        txt = r"$AR={:.2f}$".format(sec_var)
                        if var_name.lower().startswith("t"):
                            ax_txt = "T = {:.2f} K".format(var)
                        elif var_name.lower().startswith("r"):
                            ax_txt = r"$r_{{sub}}$={:.2f}$\mu m$".format(var * 1e6)
                        else:
                            ax_txt = None
                    elif secondary_var_name.lower().startswith(("p")):
                        if var > 1e-6:
                            ax_txt = r"$r_{{sub}}$={:.2f}$\mu m$".format(var * 1e6)
                        else:
                            ax_txt = r"$r_{{sub}}$={:.0f}$nm$".format(var * 1e9)

                        ax_txt = "P = {:.2f} $mW/cm^2$".format(var * 1e-1)
                    else:
                        txt = ax_txt = None
                    legend += [txt]

            else:
                y = np.abs(self.comsol.data[sel].iloc[:, col].values.flatten())
                # scale y-axis by bulk current value
                ## here it is the last one obtained
                y = y / y[-1]
                ## Plot just some data, for easier visualization
                sel2 = y >= 0.97
                y = [y[sel2]]
                x = self.comsol.data["d (m)"][sel].values.flatten()[sel2]
                # Determine offset from data.
                # This will be wrong by couple of nm if |offset - rUME| ~ 0
                if offset is None:
                    offset = self.comsol.data["d (m)"][sel].unique().min()
                x -= offset
                # Scale x-axis by pipette diameter
                x = [x / (pipette_diameter * 1e-9)]

                if var_name.lower().startswith("t"):
                    txt = "T = {:.1f} K".format(var)
                elif var_name.lower().startswith(("r")):
                    txt = r"$r_{{sub}}$={:.1f}$\mu m$".format(var * 1e6)
                else:
                    txt = None
                ax_txt = None
                legend = [txt]
                fmts = ["-k"]

            y_lab = [r"$I/I_{bulk}$"]
            x_lab = [r"$z/d$"]

            if show_experiment:
                try:
                    # Assign x-axis, set origin to 0 and scale by pipette diameter
                    x_ = self.approach.dsdata["Z(um)"]
                    x_ = x_ - np.min(x_)
                    x_ = x_ / (pipette_diameter * 1e-3)
                    x += [x_]
                    # Assign y-axis, scale by bulk value (robustly)
                    y_ = self.approach.dsdata["Current1(A)"]
                    t_ = np.cumsum(self.approach.dsdata["dt(s)"])
                    y_ = y_ / np.quantile(y_[np.nonzero(t_ < 250e-3)[0]], 0.5)
                    y += [y_]

                    # Add descriptive fields
                    legend += ["experiment"]
                    fmts += ["-gray"]
                    if i == 0:
                        title = title + "\n" + \
                                "Experiment: {} @ {}".format(self.approach.name, self.approach.date)

                    # Implement moving Aaverage on experimental data
                    if window_size > 1:
                        y_temp = pd.Series(y[-1]).rolling(window = window_size).mean()
                        y[-1] = y_temp.iloc[window_size-1:].values
                        x[-1] = x[-1][window_size-1:]
                        movav_txt = "exp., N = {:d}".format(window_size)
                        legend[-1] = movav_txt

                    # Send experimental data to the back for plotting
                    x = x[::-1]
                    y = y[::-1]
                    fmts = fmts[::-1]
                    legend = legend[::-1]

                except Exception as e:
                    # Approach most likely not assigned, don't try.
                    pass

            try:
                if add_unity_line == "y":
                    x_aux = [el1 for arr in x for el1 in arr]
                    x += [np.linspace(0, np.max(x_aux), 20)]
                    y += [np.ones_like(x[-1])]
                    fmts += [".whitesmoke"]
                    legend += ["unity line"]

                elif add_unity_line == "x":
                    y_aux = [el1 for arr in y for el1 in arr]
                    y += [np.linspace(np.min(y_aux), np.max(y_aux), 20)]
                    x += [np.ones_like(y[-1])]
                    fmts += ["-whitesmoke"]
                    legend += [r"$z/d=1$"]
            except Exception as e:
                pass # Do not add line

            plots.plot_generic(x, y, x_lab, y_lab, legend = legend, ax = axs[i],
                                fmts = fmts, text = ax_txt, **kwargs) #(0.1, 0.90)
        fig.suptitle(title, size = 10, y = 1.02 - nrows*0.02)

        fpath = os.path.join(self.comsol.datadir, self.comsol.name)
        utils.save_fig(fpath, self.comsol.date)
