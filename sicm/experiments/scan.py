import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy

from sicm import analysis, plots
from sicm.utils import utils
from sicm.io import downsample_to_linenumber
from .experiment import Experiment
from ..measurements.hops import Hops

class Scan(Experiment):
    """SCAN Object

    Parameters
    -------------
    datadir: str
        Directory where data for given experiment is stored
    exp_name: str
        Name of the experiment, must correspond to the filename
    y_trim: array-like, length two
        Tuple of (y_start, y_end), range of y-axis to preserve
    x_trim: array-like, length two
        Tuple of (x_start, x_end), range of x-axis to preserve
    do_correct: bool
        Apply correction for jump in current? Works nicely, but data is strongly influenced.
    scan_type: str
        Type of scan. ("scan", "it")
    """
    def __init__(self, datadir, exp_name, y_trim = None, x_trim = None, do_correct = False,
                scan_type = "scan"):

        super(Scan, self).__init__(datadir, exp_name, scan_type)
        self.dsdata = self._trim_data(self.dsdata, x_trim, y_trim)
        self.dsdata = self._correct_dsdata(do_correct)
        self.x_trim = x_trim
        self.y_trim = y_trim
        self._report_xy_extents()

    def _trim_data(self, data, x_trim, y_trim):
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
        if len(by) >  np.prod(X.shape)*0.1:
            factor = np.int(len(by) / (np.prod(X.shape)*0.1))
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

        self._data["time(s)"] = np.cumsum(self._data["dt(s)"])
        hop = Hops(self._data, self.idxs, self.name, self.date)
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
        x_ax = data["Z(um)"][sel] # [np.cumsum(data["dt(s)"][sel])]
        x_ax = [x_ax - np.min(x_ax)]
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

        x_lab = ["Z(um)"]# ["time(s)"]

        fpath = self.get_fpath()
        fpath = utils.make_fname(fpath, "_approach")
        plots.plot_generic(x_ax, y_ax, x_lab, y_lab, fname = fpath)


    def annotate_peaks(self, sel = None, window_size = 250):
        """todo"""
        hop = Hops(self.data, self.idxs, self.name, self.date)
        _, _ = hop.annotate_peaks(sel, window_size = window_size, save_dir = self.datadir,
                                    do_plot = True)

    def plot_slice( self, x, y, z, z_lab = "Z", ax = None, title = None,
                    fname = None, cbar_lims = (None, None)):
        """Plot a single slice"""
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            fig.tight_layout()
        else:
            fig = ax.get_figure()

        if any(cl is None for cl in cbar_lims):
            # Option 2: Variable colorbar, but better contrast for each slice
            conts = ax.tricontourf(x, y, z, cmap = "gray", levels = 10) # or greys
            cbar = fig.colorbar(conts, format = "%.4E", drawedges = True)
            # cbar.set_clim(*cbar_lims) # this appears not helpful
        else:
            # Option 1: Same colorbar for all slices
            norm = mpl.colors.Normalize(vmin = cbar_lims[0], vmax = cbar_lims[1])
            conts = ax.tricontourf(x, y, z, cmap = "gray", levels = 10, norm = norm) # or greys
            cbar = plots.make_colorbar(fig, conts.cmap, conts.levels, *cbar_lims)


        cbar.ax.set_ylabel(z_lab)

        ax.set_xlabel('X(um)')
        ax.set_ylabel('Y(um)')


        # Set descriptive title
        if title is not None:
            ax.set_title(title)
        # Save figure
        if fname is not None:
            utils.save_fig(fname, ext = ".png")

        # Explicitly close all figures if already too many;
        if len(plt.get_fignums()) > 3:
            plt.close('all')

    def plot_slices(self, tilt, n_slices = 10):
        """Plot measurmement values at different z-locations"""
        # Look at raw data
        data = self._trim_data(self._data, self.x_trim, self.y_trim)
        uniqs, cnts = np.unique(data["LineNumber"], return_counts=True)
        # Obtain all approaches
        approach_linenos = np.arange(5, max(uniqs), 3)
        ## include effect of trimming
        approach_linenos = approach_linenos[np.in1d(approach_linenos, uniqs)]
        approach_data, approach_idxs = downsample_to_linenumber(data, approach_linenos, "all")
        # obtain all retracts; retract_linenos = approach_linenos + 1
        retract_linenos = np.arange(6, max(uniqs) + 1, 3)
        ## include effect of trimming
        retract_linenos = retract_linenos[np.in1d(retract_linenos, uniqs)]
        retract_data, retract_idxs = downsample_to_linenumber(data, retract_linenos, "all")

        # Vectorization possible?
        z_axs = []
        tilt = tilt.flatten() # CHECK that correctly ordered !
        for i, (aln, rln) in enumerate(zip(approach_linenos, retract_linenos)):
            # if trimmed manually, some points will be out-of-grid and are discarded
            if i >= len(tilt):
                continue
            # Extract data for approach and following retraction at given
            # location X, Y that is descibred by i,i+1 LineNumber pair
            # discarding point at the very bottom from both sides
            z_down = approach_data["Z(um)"][approach_data["LineNumber"] == aln][:-1]
            z_up = retract_data["Z(um)"][retract_data["LineNumber"] == rln][1:]
            # Handle the up/down data as of equal nature and combine them to an array
            z = np.concatenate((z_down, z_up))
            # Remove tilt at given X,Y and set origin to the lowest location
            z = z - tilt[i]
            z = z - np.min(z)
            # Add to list of arrays
            z_axs.append(z)
        # Be conservvative and slice only where we have data for all hops.
        # This will trhow away lot of information if even a single approach
        # is much shorter than other approaches.
        # It is slightly mitigated by adding retract data, but this presumes
        # that the data from approach and retract are of comparable nature.
        # The approach is less reliable for far-away z-coordinates (>75% of z_range)
        # because occasionally only retract points will be considered for given X,Y.
        z_diffs = [zz.max() - zz.min() for zz in z_axs]
        slices_z_locs, z_delta = np.linspace(0, np.min(z_diffs), n_slices, retstep = True)
        # Adjust the interval by delta-multiplier*z_delta up
        delta_multiplier  = 0.25 # half-width of interval
        slices_z_locs, z_delta = np.linspace(delta_multiplier * z_delta,
                                slices_z_locs[-1] - delta_multiplier * z_delta, n_slices, retstep = True)

        keypairs = {"Current1(A)": r"$I_{norm} [-]$",
                    "LockinPhase": r"$\theta_{norm} [-]$"}
        slices_shape = tilt.shape + (n_slices, ) + (len(keypairs), )
        measurements = np.full(slices_shape, np.nan)
        stds = np.full_like(measurements, np.nan)

        for i, (aln, rln) in enumerate(zip(approach_linenos, retract_linenos)):
            # if trimmed manually, some points will be out-of-grid and are discarded
            if i >= len(tilt):
                continue

            for k, key in enumerate(keypairs.keys()):
                try:
                    # Pull out measurmement, discarding point at the very
                    # bottom from both sides
                    v_down = data[key][data["LineNumber"] == aln][:-1]
                    v_up = data[key][data["LineNumber"] == rln][1:]
                    v_all = np.concatenate((v_down, v_up))
                    # Normalize by value at the top of approach
                    # consider initial 250ms as period for sampling max value
                    # take almost maximum, but avoid picking an outlier
                    t = np.cumsum(data["dt(s)"][data["LineNumber"] == aln][:-1])
                    idxs = np.nonzero(t < 250e-3)[0]
                    v_all = v_all / np.quantile(v_all[idxs], 0.95)

                    for j, szl in enumerate(slices_z_locs):
                        z = z_axs[i]
                        # Look always at slice of thicknes z_delta centeted around given z
                        # Tested also slice thickness z_delta/2 but results looked alike.
                        cond1 = z >= (szl - z_delta * delta_multiplier)
                        cond2 = z < (szl + z_delta * delta_multiplier)
                        sel = np.nonzero(np.logical_and(cond1, cond2))[0]
                        v_sub = v_all[sel]

                        # Tested mean and mean, result was alike
                        # Mean may be better with noise, edian with outliers
                        measurements[i, j, k] = np.mean(v_sub)
                        stds[i, j, k] = np.std(v_sub)
                        # Warn user if data potentialy unreliable
                        if np.std(v_sub) == 0:
                            msg =   "Single datapoint available for `{}` at [{},{},{}]. " +\
                                    "Measurement is not reliable!"
                            msg = msg.format(key, i, j, k)
                            print(msg)
                except KeyError as e:
                    pass # one of the two may not be always present.

        # Pull out X, Y coordinates data
        n_points = len(tilt)
        X = np.squeeze(self.dsdata["X(um)"])[:n_points]
        Y = np.squeeze(self.dsdata["Y(um)"])[:n_points]

        # Iterate over measured parameters
        for i, (key, label) in enumerate(keypairs.items()):
            val = np.squeeze(measurements[:,:, i])

            # Skip given measurmement if we dont have data
            if np.isnan(val).all():
                continue
            # Make colorbar extents common to all slices for given key
            cbar_lims = (np.min(val), np.max(val))
            # Get path for saving
            fpath = self.get_fpath()
            fpath = utils.make_fname(fpath, subdirname = "{}/{}".format("slices", key))

            # Iterate Over Slices
            for slice_id in np.arange(0, val.shape[-1]):
                slice = np.squeeze(val[:, slice_id])
                z_annot = str(np.around(slices_z_locs[slice_id], 4))
                title = "slice {:d} @ z = {} um".format(slice_id, z_annot)
                fname = utils.make_fname(fpath, "_" + str(np.int(slice_id)))
                self.plot_slice(X, Y, slice, label,
                                title = title, fname = fname,
                                cbar_lims = cbar_lims)

    def plot_surface(self, plot_current = False, plot_slices = False):
        """Plot surface as contours and 3D"""
        # Plot downsampled Data
        result = self.dsdata
        X = np.squeeze(result["X(um)"])
        Y = np.squeeze(result["Y(um)"])

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

        elif plot_current and not plot_slices:
            # if you want to plot slices, allow leveling Z
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

        if plot_slices:
            self.plot_slices(Z_tilt)
        else:
            plt.style.use("seaborn")
            fig = plt.figure(figsize = (12, 10))
            # fig.tight_layout()

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
            ax.plot_trisurf(X[:npoints], Y[:npoints], Z_sq.flatten(), cmap='binary')
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

            # Save figure
            fpath = self.get_fpath()
            fname = utils.make_fname(fpath, "_surface")
            utils.save_fig(fname, ext = ".png")
            # Show
            plt.show()
