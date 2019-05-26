import os
import numpy as np
import timeit
from scipy.optimize import curve_fit


from sicm import io
from sicm.plots import plots
from sicm.utils import utils
from .model import Model
from sicm.sicm import Approach, ApproachList


import matplotlib.pyplot as plt

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
            return variable
        else:
            keys = list(self.comsol.data.keys())
            n_repeats = [len(self.comsol.data[k].unique()) for k in keys]
            valid_keys  = [k for k, nr in zip(keys, n_repeats) if nr > 1 and k != "d (m)"]
            return valid_keys[0]

    def plot_grid(self, pipette_diameter = 220, show_experiment = True, variable = None,
                    col = 2, offset = 0, add_unity_line = False):
        """Plot all numerical results on grid

        Displays result of numerically simulated approach at different temperatures.
        If experimental data avialable it is superimposed in all plots.
        """
        var_name = self._select_variable(variable)
        try:
            uniqs = self.comsol.data[var_name].unique()
        except KeyError as e:
            uniqs = [None]
        nrows = np.int(np.ceil(len(uniqs) / 2))
        fig, axs = plt.subplots(nrows = nrows, ncols = 2, figsize = (10, nrows * 4))
        axs = axs.flatten()

        title = "Comsol Param Grid" + "\n" + \
                "Model: {} @ {}".format(self.comsol.name, self.comsol.date)

        for i, var in enumerate(uniqs):
            # Look at given temperature
            if var is None:
                sel = [True] * len(self.comsol.data)
            else:
                sel = self.comsol.data[var_name] == var
            y = np.abs(self.comsol.data[sel].iloc[:, col].values.flatten())
            # scale y-axis by bulk current value
            ## here it is the last one obtained
            y = y / y[-1]
            ## Plot just some data, for easier visualization
            sel2 = y >= 0.97
            y = [y[sel2]]
            x = self.comsol.data["d (m)"][sel].values.flatten()[sel2]
            # Scale x-axis by pipette diameter
            x = [x / (pipette_diameter * 1e-9)]

            if var_name.lower().startswith("t"):
                txt = "T = {:.1f} K".format(var)
            elif var_name.lower().startswith(("r")):
                txt = r"$r_{{sub}}$={:.1f}$\mu m$".format(var * 1e6)
            legend = [txt]
            y_lab = [r"$I/I_{bulk}$"]
            x_lab = [r"$z/d$"]
            fmts = ["-k"]

            if show_experiment:
                try:
                    # Assign x-axis, set origin to 0 and scale by pipette diameter
                    x_ = self.approach.dsdata["Z(um)"]
                    x_ = x_ - np.min(x_)
                    if offset is None:
                        offset = self.comsol.data["d (m)"].unique().min() * 1e6
                    x_ += offset
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
                except Exception as e:
                    # Approach most likely not assigned, don't try.
                    pass

            if add_unity_line:
                x += [np.linspace(0, np.max(x[0]), 20)]
                y += [np.ones_like(x[-1])]
                fmts += [".whitesmoke"]
                legend += ["unity line"]

            plots.plot_generic(x, y, x_lab, y_lab, legend = legend, ax = axs[i],
                                fmts = fmts)
        fig.suptitle(title, size = 12, y = 0.92)

        fpath = os.path.join(self.comsol.datadir, self.comsol.name)
        utils.save_fig(fpath, self.comsol.date)
