import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from sicm import analysis, plots
from sicm.utils import utils
from .experiment import Experiment
from ..measurements.hops import Hops

class Scan(Experiment):
    """SCAN Object

    TODO:
        - allow downsampling if too many datapoints

    Parameters
    -------------

    """
    def __init__(self, datadir, exp_name, y_trim = None, x_trim = None, do_correct = False,
                scan_type = "scan"):

        super(Scan, self).__init__(datadir, exp_name, scan_type)
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


    def _downsample_surface_data(self, X, Y, Z, by = 10):
        """Downsamples data available for plotting by factor `by`

        When plotting scan constant_distance, the data will be plentiful.
        To ease visualization, we downsample them.
        """
        if not isinstance(by, (list, tuple, np.ndarray)):
            by = np.arange(0, np.prod(X.shape), by)
        if len(by) >  np.prod(X.shape)*0.05:
            factor = np.int(len(by) / (np.prod(X.shape)*0.05))
            by = np.arange(0, np.prod(X.shape), factor)
        print(  "Downsampled from {} to {} datapoints for `plot_surface`.".\
                format(np.prod(X.shape), len(by)))

        return X[by], Y[by], Z[by]


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
        hop.plot(sel, fname = fpath, do_annotate = not self.is_it)

    def _plot_approach(self):
        """Plots approach of a scan

        Normally, we don't care about an approach of a scan, but occasionaly
        user may want to interrogate it.
        """
        # extract approach from raw data
        data = self._data
        uniqs, cnts = np.unique(data["LineNumber"], return_counts=True)
        # it is the longest one or the second one or "2", any works
        # hardcodding 2 may be the most robust!
        approach_lineno = uniqs[np.argmax(cnts)]
        sel = data["LineNumber"] == approach_lineno
        x_ax = [np.cumsum(data["dt(s)"][sel])]
        y_ax = []
        y_lab = []

        try:
            y_ax.append(data["Current1(A)"][sel])
            y_lab.append("Current1(A)")
        except KeyError as e:
            pass

        try:
            y_ax.append(data["LockinPhase"][sel])
            y_lab.append("LockinPhase")
        except KeyError as e:
            pass

        x_lab = ["time(s)"]

        fpath = self.get_fpath()
        fpath = utils.make_fname(fpath, "_approach")
        plots.plot_generic(x_ax, y_ax, x_lab, y_lab, fname = fpath)


    def annotate_peaks(self, sel = None, window_size = 250):
        """todo"""
        hop = Hops(self.data, self.idxs, self.name, self.date)
        _, _ = hop.annotate_peaks(sel, window_size = window_size, save_dir = self.datadir,
                                    do_plot = True)

    def plot_slice(self, x, y, z, z_lab = "Z", ax = None):
        """Plot a Single Slice"""
        if ax is None:
            _, ax = plt.subplot(1, 1)

        C = ax.tricontourf(x, y, z, cmap='viridis')
        CB = fig.colorbar(C)
        ax.set_xlabel('X(um)')
        ax.set_ylabel('Y(um)')
        ax.set_title(z_lab)

    def plot_slices(self, tilt, n_slices = 10):
        """Plot measurmement values at different z-locations"""
        import pdb; pdb.set_trace()
        # Pull out whole approach curve for each hop
        data = self.data
        linenos, cnts = np.unique(data["LineNumber"], return_counts = True)

        # Vectorization possible?
        z_axs = []
        measurmements = []
        tilt = tilt.flatten() # CHECK that correctly ordered
        for i, ln in enumerate(linenos):
            z = data["Z(um)"][data["LineNumber"] == ln]
            tilt_temp = np.asarray([tilt[i]] * cnts[i])
            z = z - (tilt_temp - np.min(tilt_temp))
            z_axs.append(z)

            nested_measurements = []
            for k in ["Current1(A)", "LockinPhase"]:
                try:
                    nested_measurmements.append(data[k][data["LineNumber"] == ln])
                except KeyError as e:
                    pass # one of te two may not be always present.
            measurements.append(nested_measurements)

        # z_axs = np.asarray(z_axs)
        # measurmemnts = np.asarray(measurmemnts)

        z_min = np.min(z) # may need to cast to array
        z_max = np.max(z)
        z_range = np.linspace(z_min, z_max, n_slices)
        z_delta = (np.max(z_range) - np.min(z_range)) / (n_slices*5)

        for i, z, vals in enumerate(zip(z_axs, measurmements)):

    def plot_surface(self, plot_current = False):
        """Plot surface as contours and 3D"""
        # Plot downsampled Data
        result = self.dsdata
        X = np.squeeze(result["X(um)"])
        Y = np.squeeze(result["Y(um)"])

        # Plot slices at various Zs?
        plot_slices = False
        if self.is_constant_distance:
            # We care about measurements of current
            Z = np.squeeze(result["Current1(A)"])
            z_lab = "Current1(A)"
            # Pick current values with consistent coordinate only
            # This works as long as the movemement in Z is just due to noise
            Z_aux = np.squeeze(result["Z(um)"])
            uniqs, cnts = np.unique(Z_aux, return_counts = True)
            to_keep = np.nonzero(Z_aux == uniqs[np.argmax(cnts)])[0]
            # Note that downsampling is stronger for most datasets! (see the function body)
            X, Y, Z = self._downsample_surface_data(X, Y, Z, to_keep)

        elif plot_current:
            # if plotting current on z-axis, it is preferable to obtain data
            # at various z-levels and possibly average over slice of thicnkes 2*\delta
            plot_slices = True
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
        X_sq, Y_sq, Z_sq, Z_tilt = analysis.level_plane(X, Y, Z, True, is_interactive,
                                                        z_lab = z_lab)
        npoints = len(Z_sq.flatten())

        plt.style.use("seaborn")
        fig = plt.figure(figsize = (12, 10))
        fig.tight_layout()

        # Filled countour with triangulation

        ax = fig.add_subplot(2, 2, 1)
        self.plot_slice(X[:npoints], Y[:npoints], Z_sq.flatten(), z_lab, ax)
        # C = ax.tricontourf(X[:npoints], Y[:npoints], Z_sq.flatten(), cmap='viridis')
        # CB = fig.colorbar(C)
        # ax.set_xlabel('X(um)')
        # ax.set_ylabel('Y(um)')
        # ax.set_title(z_lab)

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
