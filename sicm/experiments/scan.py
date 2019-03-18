import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from sicm import analysis, plots
from .experiment import Experiment
from ..measurements.hops import Hops


class Scan(Experiment):
    """SCAN Object

    TODO:
        - allow downsampling if too many datapoints
    """
    def __init__(self, datadir, exp_name, x_trim = None, y_trim = None, do_correct = False,
                is_constant_distance = False):
        super(Scan, self).__init__(datadir, exp_name, is_constant_distance)
        self.dsdata = self._trim_dsdata(x_trim, y_trim)
        self.dsdata = self._correct_dsdata(do_correct)
        self._report_xy_extents()

    def _trim_dsdata(self, x_trim, y_trim):
        data = self.dsdata
        if not y_trim: y_trim = None
        if not x_trim: x_trim = None

        if y_trim or x_trim is not None:
            X = np.squeeze(data["X(um)"])
            Y = np.squeeze(data["Y(um)"])

            if x_trim is None:
                keep_idx = [True]*len(X)
            else:
                keep_idx = np.logical_and(X > x_trim[0], X <= x_trim[-1])
            if y_trim is None:
                keep_idy = [True]*len(Y)
            else:
                keep_idy = np.logical_and(Y > y_trim[0], Y <= y_trim[-1])

            keep_id = np.logical_and(keep_idx, keep_idy)

            trim_data = deepcopy(data)
            for k,v in data.items():
                trim_data[k] = v[keep_id]

            return trim_data
        else:
            return data

    def _correct_dsdata(self, do_correct):
        if do_correct:
            result = analysis.correct_for_current(self.dsdata)
            return result
        else:
            return self.dsdata

    def _report_xy_extents(self):
        """Report extents on XY axis

        Assumes that superclass assigns property _data, which holds all data
        obtained from TSV files.
        """
        print("Veryfying X, Y extents:")
        print("xmax: {}, xmin: {},\nymax: {}, ymin: {}\nxdiff: {}, ydiff: {}".\
              format(
                    self._data["X(um)"].max(),
                    self._data["X(um)"].min(),
                    self._data["Y(um)"].max(),
                    self._data["Y(um)"].min(),
                    self._data["X(um)"].max() - self._data["X(um)"].min(),
                    self._data["Y(um)"].max() - self._data["Y(um)"].min()))


    def plot_hopping_scan(self, sel = None):
        """Plot results of hopping scan

        If data is aquired with QTF setup, voltage and current are not available.
        """
        exp_name  = self.name
        date = self.date
        plots.plot_sicm(self.dsdata, sel, "Hopping Scan", exp_name, date)

    def plot_hops(self, sel = None, do_save = True):
        """todo"""
        if do_save:
            fpath = self.get_fpath()
        else:
            fpath = None

        hop = Hops(self.data, self.idxs, self.name, self.date)
        hop.plot(sel, fname = fpath)

    def annotate_peaks(self, sel = None, window_size = 250):
        """todo"""
        hop = Hops(self.data, self.idxs, self.name, self.date)
        _, _ = hop.annotate_peaks(sel, window_size = window_size, save_dir = self.datadir,
                                    do_plot = True)

    def plot_surface(self, plot_current = False):
        """Plot surface as contours and 3D"""

        result = self.dsdata
        X = np.squeeze(result["X(um)"])
        Y = np.squeeze(result["Y(um)"])
        if self.is_constant_distance or plot_current:
            Z = np.squeeze(result["Current1(A)"])
            z_lab = "Current1(A)"
        else:
            Z = np.squeeze(result["Z(um)"])
            z_lab = "Z(um)"
        # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
        try:
            if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
                is_interactive = False # jupyter
            else:
                is_interactive = True # ipython
        except NameError as e:
            is_interactive = True # command line
        # Level Z coordinates and convert to matrix with proper ordering
        X_sq, Y_sq, Z_sq = analysis.level_plane(X, Y, Z, True, is_interactive, z_lab = z_lab)
        npoints = len(Z_sq.flatten())

        plt.style.use("seaborn")
        fig = plt.figure(figsize = (12, 10))
        fig.tight_layout()

        # Filled countour with triangulation
        ax = fig.add_subplot(2, 2, 1)
        C = ax.tricontourf(X[:npoints], Y[:npoints], Z_sq.flatten(), cmap='viridis')
        CB = fig.colorbar(C)
        ax.set_xlabel('X(um)')
        ax.set_ylabel('Y(um)')
        ax.set_title(z_lab)

        # Surface in 3D projection
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.plot_trisurf(X[:npoints], Y[:npoints], Z_sq.flatten(), cmap='viridis')
        ax.set_xlabel('X(um)')
        ax.set_ylabel('Y(um)')
        ax.set_zlabel(z_lab)

        # # Filled contours without triangulation
        # ax = fig.add_subplot(2, 2, 3)
        # C = ax.contourf(X_sq, Y_sq, Z_sq, cmap='viridis')
        # CB = fig.colorbar(C)
        # ax.set_xlabel('X(um)')
        # ax.set_ylabel('Y(um)')
        # ax.set_title('Z(um)')

        plt.show()
