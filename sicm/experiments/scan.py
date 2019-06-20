import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy
from collections import Counter
from scipy.stats import trim_mean

from sicm import analysis
from sicm.plots import plots, surface
from sicm.utils import utils
from sicm.io import downsample_to_linenumber
from .experiment import Experiment
from ..measurements.hops import Hops
from sicm.mathops.conversion import convert_measurements
from sicm.mathops import mathops

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
    def __init__(   self, datadir, exp_name, y_trim = None, x_trim = None,
                    do_correct = False, convert = False, scan_type = "scan"):
        super(Scan, self).__init__(datadir, exp_name, scan_type)
        self.dsdata = self._trim_data(self.dsdata, x_trim, y_trim)
        self.dsdata = self._correct_dsdata(do_correct)
        self.x_trim = x_trim
        self.y_trim = y_trim
        self.convert = convert
        self._report_xy_extents()

    def _trim_data(self, data, x_trim, y_trim):
        """Trim Data to x, y range

        Parameters
        ---------
        x_trim: tuple
            (low, high) limits for x coordinate to preseve.
        y_trim: tuple
            (low, high) limits for y coordinate to preseve.

        Returns
        ----------
        trim_data:
            Data, trimmed if any of x_trim, y_trim is not None.

        """
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
        """Applies correction for jump of current

        Use only for visualization purposes!!!
        """

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

    def _calculate_step_size(self):
        """Calculate Step Size"""
        uniqs, cnts = np.unique(self.dsdata["LineNumber"], return_counts = True)
        x_diffs = np.diff(self.dsdata["X(um)"].flatten())
        y_diffs = np.diff(self.dsdata["Y(um)"].flatten())

        # https://stackoverflow.com/q/2652368
        # y_idx = np.argwhere(np.sign(x_diffs[:-1]) != np.sign(x_diffs[1:])).flatten() + 1
        y_step = np.max(np.abs(y_diffs))
        x_step = np.sort(np.abs(x_diffs))[-2]

        return x_step, y_step

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

    def _aggregate_surface_data(self, X, Y, Z):
        """Aggregate data for X, Y coordinates that are within noise level

        Parameters
        --------------
        X, Y, Z: array-like
            X, Y coordinates of measurements Z

        Returns
        ------------------
        X_, Y_, Z_: array-like
            x, y coordinates of measurements z
        """
        # Measurements are grouped together for y_axis by linenumber
        linenos = np.squeeze(self.dsdata["LineNumber"])
        uniqs, cnts = np.unique(linenos, return_counts=True)
        n_lines = len(uniqs)

        X_ = []
        Y_ = []
        Z_ = []
        for n in uniqs:
            n_mask = np.nonzero(linenos == n)[0]
            # take most common value for Y, median ,mean should work equally well
            ## Elements with equal counts are ordered arbitrarily!
            line_y = Counter(Y[n_mask]).most_common(1)[0][0]
            line_x = X[n_mask]
            line_z = Z[n_mask]
            # Take some arbitrary number of points on X axis
            ## x-axis is linearly increasing so take enough points!
            try:
                split_ids = np.linspace(0, len(line_x), 2*n_lines + 1, dtype = np.int)
                assert len(split_ids) == len(set(split_ids))
            except AssertionError as e: # should never happen actually
                split_ids = np.sort(np.unique(split_ids))

            # Split to `n_lines` intervals
            line_x = np.split(line_x, split_ids[1:-1])
            line_z = np.split(line_z, split_ids[1:-1])
            # Describe each interval by mean
            ## median / trim_mean are alternatives
            line_x = [np.mean(x) for x in line_x]
            line_z = [np.mean(z) for z in line_z]
            # Replicate y-value as needed
            line_y = [line_y] * len(line_x)
            X_.extend(line_x); Y_.extend(line_y); Z_.extend(line_z)

        # Make sure arrays of same length
        assert len(set(len(i) for i in [X_, Y_, Z_])) == 1

        return np.asarray(X_), np.asarray(Y_), np.asarray(Z_)

    def _select_approach(self, data, location = None):
        """Select approach

        By default selects the very first approach to surface.

        Paramters
        --------------
        location: None or tuple
            If tuple, then (X, Y) coordinates of approach to select

        Returns
        -------------------
        sel: array-like
            Bool array indicating data to select.
        """
        if location is None:
            uniqs, cnts = np.unique(data["LineNumber"], return_counts=True)
            # it is the longest one or the second one or "2", any works
            # hardcodding 2 may be the most robust!
            approach_lineno = uniqs[np.argmax(cnts)]
            sel = data["LineNumber"] == approach_lineno
            loc_str = ""
        else:
            dsdata = self.dsdata # work with downsampled data
            x, y = location
            x_step, y_step = self._calculate_step_size()
            x_sel = np.logical_and( dsdata["X(um)"] > x - x_step / 3,
                                    dsdata["X(um)"] < x + x_step / 3)
            y_sel = np.logical_and( dsdata["Y(um)"] > y - y_step / 3,
                                    dsdata["Y(um)"] < y + y_step / 3)
            xy_sel = np.logical_and(x_sel, y_sel)
            approach_lineno = np.unique(dsdata["LineNumber"][xy_sel])
            if len(approach_lineno) > 1:
                msg =   "Found {} approaches in selected location. Picking one randomly."\
                        .format(np.int(len(approach_lineno)))
                print(msg)
                approach_lineno = np.random.choice(approach_lineno)
            else:
                approach_lineno = approach_lineno[0]
            sel = data["LineNumber"] == approach_lineno


        x_loc = np.median(data["X(um)"][sel])
        y_loc = np.median(data["Y(um)"][sel])
        if location is not None:
            loc_str = "X{}Y{}".format(  np.str(np.around(x_loc, 3)).replace(".", "p"),
                                        np.str(np.around(y_loc, 3)).replace(".", "p"))

            msg = "Selected approach @ X:{:.3f}, Y:{:.3f} um".format(x_loc, y_loc)
            print(msg)
        return sel, loc_str

    def _remove_last_datapoint(self, X, Y, Z):
        # Remove last point if the data was not trimmed.
        was_trimmed = any([False if x is None else True for x in (self.y_trim, self.y_trim)])
        if not self.is_constant_distance and not was_trimmed:
            X, Y, Z = X[:-1], Y[:-1], Z[:-1]
        return X, Y, Z

    def _calculate_interval_adjustment(approach_linenos, approach_data):
        """Calculate length-of-interval adjustment for first approach

        ???: This is most-likely not going to be used anymore. Remove on next revision.

        Attempt to figure out how to adjust a first approach that is usually much
        longer than other.

        Returns
        -----------
        first_length_adj: int
            The index to data, from end, that marks interval to keep (e.g.
            z_down = z_down[-first_length_adj:])
        """
        # Gather information on approach/retract speeds.
        speeds = np.empty((len(approach_linenos), 2))
        lengths = np.empty((len(approach_linenos), 2))
        for i, (aln, rln) in enumerate(zip(approach_data, approach_linenos)):
            z_down = approach_data["Z(um)"][approach_data["LineNumber"] == aln]
            #z_up = retract_data["Z(um)"][retract_data["LineNumber"] == rln]
            dt_down = approach_data["dt(s)"][approach_data["LineNumber"] == aln][1:]
            #dt_up = retract_data["dt(s)"][retract_data["LineNumber"] == rln][1:]
            ## Intial about 150 points of approach are collected while z is still constant!
            ## This happends while X piezo readjusts, here we manually adjust for this
            z_diff_down = np.abs(np.diff(z_down))
            try:
                z_step_down = np.sort(z_diff_down)[-5]
            except IndexError as e:
                z_step_down = np.quantile(z_diff_down, 0.9)

            after_init_down = np.nonzero(z_diff_down > z_step_down / 3)[0]
            down_speed = np.mean(z_diff_down[after_init_down] / dt_down[after_init_down])
            ## Speed of movement up
            up_speed = np.mean(np.abs(np.diff(z_up) / dt_up))
            ## Assing.
            speeds[i, :] = [down_speed, up_speed]
            lengths[i, :] = [len(z_down), len(z_up)]

        # Compute interval length adjustment for the first approach
        speed_ratio = np.mean(speeds[1:, 0]) / speeds[0, 0]
        first_length_adj = np.int(np.mean(lengths[1:, 0]) * speed_ratio)
        return first_length_adj

    def annotate_peaks(self, sel = None, window_size = 250):
        """Mark locations of low-peaks in data

        Low peaks correspond to datapoints acquired closest to surface.

        Parameters
        --------
        sel: array-like
        window_size: int
            Width of range in which to look for peak.
        """
        hop = Hops(self.data, self.idxs, self.name, self.date)
        _, _ = hop.annotate_peaks(sel, window_size = window_size, save_dir = self.datadir,
                                    do_plot = True)

    def plot_hopping_scan(self, sel = None, mark_points = False, do_save = True):
        """Plot results of hopping scan

        If data is aquired with QTF setup, voltage and current are not available.

        Parameters
        -------------------
        sel: array-like
        mark_points: bool
            Should the bottom points of each approach be labelled?
        do_save: bool
        """
        exp_name  = self.name
        date = self.date
        plot_data = self._data

        if mark_points: # Show lowest point of approach
            if not self.is_it or not self.is_constant_distance:
                desired_linenos = self.dsdata["LineNumber"].flatten()
            else:
                msg = "Marking Points can be done only for `scan` experiment-type."
                raise NotImplementedError(msg)

            all_linenos = plot_data["LineNumber"]
            target_idx = []
            for dl in desired_linenos:
                # Take the last point of each approach from plot_data
                target_idx.append(np.max(np.argwhere(all_linenos == dl)))
            idxs = np.asarray(target_idx)
        else:
            idxs = None
            plot_data = self.dsdata

        # Saving data
        if do_save:
            fpath = self.get_fpath()
            fname = utils.make_fname(fpath, "_sicmPlot", ext = ".png")
        else:
            fname = None

        plots.plot_sicm(plot_data, sel, "Hopping Scan", exp_name, date, fname, idxs)

    def plot_hops(self, sel = None, do_save = True):
        """Plot approach curves

        Parameters
        ---------
        sel: tuple
            Range on time-axis to plot
        do_save: bool
            Should the figure be saved?
        """
        if do_save:
            fpath = self.get_fpath()
        else:
            fpath = None

        data = self._data
        if "time(s)" not in data.keys():
            data["time(s)"] = np.cumsum(data["dt(s)"])

        hop = Hops(data, self._idxs, self.name, self.date)
        hop.plot(sel, fname = fpath, do_annotate = not self.is_it)

    def plot_approach(self, location = None, relative = True):
        """Plots approach of a scan

        Normally, we don't care about an approach of a scan, but occasionaly
        user may want to interrogate it.

        Parameters
        -----------
        location: None or tuple
            If tuple, then (X, Y) coordinates of approach to select, else selects
            the very first approach.
        """
        # extract approach from raw data
        data = self._data
        sel, loc_str = self._select_approach(data, location)

        x_ax = data["Z(um)"][sel] # [np.cumsum(data["dt(s)"][sel])]
        x_ax = [x_ax - np.min(x_ax)]
        y_ax = []
        y_lab = []
        t =  np.cumsum(data["dt(s)"][sel])

        try:
            y = data["Current1(A)"][sel]*1e9
            if relative:
                y , _= mathops.scale_by_init(t, y)
                y_lab.append(r"$I_{norm}\ (-)$")
            else:
                y_lab.append(r"$I\ (nA)$")
            y_ax.append(y)
        except KeyError as e:
            pass

        try:
            y = data["LockinPhase"][sel]
            y = data["LockinAmplitude"][sel]
            if relative:
                y , _= mathops.scale_by_init(t, y)
                # y_lab.append(r"$\theta_{norm}$")
                y_lab.append(r"$V_{rel}\ (-)$")
            else:
                y_lab.append(r"$\theta\ (\degree)$")
            y_ax.append(y)
        except KeyError as e:
            pass

        x_lab = [r"$Z\ (um)$"]# ["time(s)"]
        fpath = self.get_fpath()
        fpath = utils.make_fname(fpath, "_approach{}".format(loc_str), ext = ".png")
        plots.plot_generic(x_ax, y_ax, x_lab, y_lab, fname = fpath)

    def plot_slices(self, X, Y, tilt, n_slices = 10, thickness = .9,
                    zrange = (None, None), clip = (None, None),
                    scaleby = "hop", center = True,  n_levels = 10,
                    descriptor = "mean", Z_aux = None, adjust = False,
                    z_coords = "common"):
        """Plot measurmement values at different z-locations as slices

        Parameters
        ---------------
        X: array-like
        Y: array-like
        tilt: array_like
        n_slices: int
            Number of slices to render
        z_coords: str
            Either "common" or "all". Choosing "common" selects maximum z such that
            it is present in all approaches/retracts.
        scaleby: str
            Either "hop" or "bulk". Choosing "hop" scales each rendered value by
            the value at the top of corresponding approach.
        center: bool
            Center x,y around zero?
        Z_aux: np.ndarray or None
            If Z_aux is an array, it is overlayed as line-contours
        """
        ## Look at raw data and trim here - OLD VERSION
        # data = self._trim_data(self._data, self.x_trim, self.y_trim)
        # uniqs, cnts = np.unique(data["LineNumber"], return_counts=True)

        ## Get infromation from dsdata and use nontrimmed raw data
        trimmed_data = self.dsdata
        data = self._data
        uniqs, cnts = np.unique(trimmed_data["LineNumber"], return_counts=True)
        # Obtain all approaches
        approach_linenos = np.arange(2, max(uniqs) -1, 3)
        ## include effect of trimming
        approach_linenos = approach_linenos[np.in1d(approach_linenos, uniqs)]
        approach_data, approach_idxs,_ = downsample_to_linenumber(data, approach_linenos, "all")
        # obtain all retracts
        retract_linenos = approach_linenos + 1
        ## include effect of trimming
        retract_linenos = retract_linenos[np.in1d(retract_linenos, uniqs + 1)]
        retract_data, retract_idxs,_ = downsample_to_linenumber(data, retract_linenos, "all")

        # extract bulk current from intial approach from raw data
        if scaleby.lower().startswith("b"):
            data_all = self._data
            sel, _ = self._select_approach(data_all, None)
            approach_current = data_all["Current1(A)"][sel]
            approach_t = np.cumsum(data_all["dt(s)"][sel])
            x, y = mathops.rolling_mean(approach_t, approach_current, 10)
            _, bulk_current = mathops.scale_by_init(x, y,"trim", 0.1)

        # Vectorization possible?
        z_axs = []
        z_surfs = []

        for i, (aln, rln) in enumerate(zip(approach_linenos, retract_linenos)):
            # Extract data for approach and following retraction at given
            # location X, Y that is descibred by i,i+1 LineNumber pair
            # discarding (very variable) point at the very bottom from both sides
            z_down = approach_data["Z(um)"][approach_data["LineNumber"] == aln]
            z_up = retract_data["Z(um)"][retract_data["LineNumber"] == rln]

            # Look only at the part of data that corresponds to roughly same distance
            # from the surface as other datapoints would
            if i == 0:
                first_app_keep = np.nonzero(z_down <= np.max(z_up))[0]
                z_down = z_down[first_app_keep]
                # z_down = z_down[-first_length_adj:]

            # Handle the up/down data as of equal nature and combine them to an array
            z = np.concatenate((z_down[:-1], z_up[1:]))
            # Remove tilt at given X,Y and set origin to the lowest location
            z = z - tilt[i]
            z_min = np.min(z)
            z = z - z_min
            # Add to list of arrays
            z_axs.append(z)
            ## keep surface values, just for interest
            z_surf = np.asarray([z_down[-1], z_up[1]])
            z_surf = z_surf - tilt[i]
            # z_surf = z_surf - z_min
            # Add to list
            z_surfs.append(np.mean(z_surf))

        # Be conservvative and slice only where we have data for all hops.
        # This will trhow away lot of information if even a single approach
        # is much shorter than other approaches.
        # It is slightly mitigated by adding retract data, but this presumes
        # that the data from approach and retract are of comparable nature.
        # The approach is less reliable for far-away z-coordinates (>75% of z_coords)
        z_diffs = [zz.max() - zz.min() for zz in z_axs]
        if z_coords.lower().startswith("c"):
            interval_end = np.min(z_diffs)
        elif z_coords.lower().startswith("a"):
            # interval_end = np.max(z_diffs)
            raise NotImplementedError("Sparse measurements are not implemented!")
        else:
            raise NotImplementedError("Only `z_coords='common'` is implemented!")

        if zrange is None:
            zrange = (0, interval_end)
        else:
            assert zrange[0] >= 0
        slices_z_locs, z_delta = np.linspace(   zrange[0], zrange[1], n_slices,
                                                retstep = True)
        # Make sure slcies have whole thicknes within intervall boundaries.
        delta_multiplier  = thickness / 2 # relative half-width of interval
        slices_z_locs, z_delta = np.linspace(slices_z_locs[0]+delta_multiplier*z_delta,
                                slices_z_locs[-1] - delta_multiplier * z_delta, n_slices,
                                retstep = True)
        # Keep the slice thickness for further use as string in annotation
        thickness_annot = str(np.around(z_delta * delta_multiplier * 2, 4))

        keypairs = {"Current1(A)": r"$I_{norm} [-]$",
                    "LockinPhase": r"$\theta_{norm} [-]$"}
        slices_shape = tilt.shape + (n_slices, ) + (len(keypairs), )
        measurements = np.full(slices_shape, np.nan)
        stds = np.full_like(measurements, np.nan)
        v_surfs = np.full(tilt.shape + (len(keypairs), ), np.nan)
        good_keys = []

        ## Obtain descriptor
        descriptor_fun = mathops.get_descriptor_fun(descriptor)
        for i, (aln, rln) in enumerate(zip(approach_linenos, retract_linenos)):
            for k, key in enumerate(keypairs.keys()):
                # one of the two may not be always present.
                if key not in data.keys():
                    continue
                else:
                    good_keys.append(key)
                # Pull out measurmement
                v_down = data[key][data["LineNumber"] == aln]
                v_up = data[key][data["LineNumber"] == rln]
                if i == 0:
                    v_down = v_down[first_app_keep]
                    # v_down = v_down[-first_length_adj:] ## --- OLD VERSION

                # Concatenate discarding point at the very bottom from both sides
                v_all = np.concatenate((v_down[:-1], v_up[1:]))

                if scaleby.lower().startswith("b"):
                    scaler = bulk_current
                elif scaleby.lower().startswith("inf"):
                    # This may work if you sample long enough distance of approach
                    # Such that it faithfully represents an exponential
                    # Note that size of window for rolling mean should be optimzed.
                    # shift = np.int(len(v_down)//7)
                    # x = np.arange(len(v_down)-(shift - 1), 1, -1)
                    # x, y = mathops.rolling_mean(x, v_down[:-shift], 125)
                    # scaler = mathops.find_bulk_val(x, y)
                    raise NotImplementedError("Scalling by value at infinity is not implemented!")
                else:
                    # Normalize by value at the top of approach
                    # consider initial 250ms as period for sampling max value
                    # take almost maximum, but avoid picking an outlier
                    t = data["dt(s)"][data["LineNumber"] == aln]
                    if i == 0:
                        t = t[first_app_keep]
                    x, y = mathops.rolling_mean(np.cumsum(t), v_down, 15)
                    _, scaler = mathops.scale_by_init(x, y, q = 0.05, t_len = 150e-3)
                    # _, scaler = mathops.scale_by_init(np.cumsum(t), v_all)

                v_all = v_all / scaler

                ## Preserve surface values, just for interest
                v_surf = np.mean(np.asarray([v_down[-1], v_up[1]])) / scaler

                ## Apply conversion if asked for
                if self.convert:
                    v_all = convert_measurements(v_all)
                    v_surf = convert_measurements(v_surf)

                v_surfs[i, k] = v_surf

                for j, szl in enumerate(slices_z_locs):
                    z = z_axs[i]
                    # Look always at slice of thicknes z_delta centeted around given z
                    # Tested also slice thickness z_delta/2 but results looked alike.
                    cond1 = z >= (szl - z_delta * delta_multiplier)
                    cond2 = z < (szl + z_delta * delta_multiplier)
                    sel = np.nonzero(np.logical_and(cond1, cond2))[0]
                    if len(sel) > 0:
                        v_sub = v_all[sel]

                        # Tested mean and mean, result was alike
                        # Mean may be better with noise, edian with outliers
                        measurements[i, j, k] = descriptor_fun(v_sub)
                        stds[i, j, k] = np.std(v_sub)
                        # Warn user if data potentialy unreliable
                        if np.std(v_sub) == 0:
                            msg =   "Single datapoint available for `{}` at [{},{},{}]. " +\
                                    "Measurement is not reliable!"
                            msg = msg.format(key, i, j, k)
                            print(msg)
                    else:
                        measurements[i, j, k] = np.nan

        # Iterate over measured parameters

        for i, (key, label) in enumerate(keypairs.items()):
            val_ = np.squeeze(measurements[:,:, i])
            val_stds = np.squeeze(stds[:,:, i])
            # Skip given measurmement if we dont have data
            if key not in good_keys: continue
            if np.isnan(val_).all():
                raise Exception("All measurements for {} are NaN!".format(key))

            if adjust:
                val, params_adj = surface.adjust_saliency(deepcopy(val_), clip, X, Y)
            else:
                val = val_
                params_adj = {}

            if self.convert: label = r"$\Delta T [K]$"
            # Make colorbar extents common to all slices for given key

            cbar_lims_stds  = (np.nanmin(val_stds), np.nanmax(val_stds))
            cbar_lims = (np.nanmin(val), np.nanmax(val))
            # Apply clipping for visualization if desired
            if clip is not None:
                opt = 2
                if opt == 1: # Clip data for plotting
                    val[val > clip[1]] = clip[1]
                    val[val < clip[0]] = clip[0]
                    cbar_lims = (None, None)
                elif opt == 2: # Let clipping happen in plotting function
                    if not isinstance(clip, (tuple, )): clip = tuple(clip)
                    cbar_lims = clip
                else:
                    raise NotImplementedError
            # Get path for saving
            fpath_ = self.get_fpath()
            fpath = utils.make_fname(fpath_, subdirname = "{}/{}".format("slices", key))
            fpath_stds = utils.make_fname(fpath_, subdirname = "{}/{}/{}".format("slices", key, "stds"))

            # Iterate Over Slices
            for slice_id in np.arange(0, val.shape[-1]):
                slice = np.squeeze(val[:, slice_id])
                slice_stds = np.squeeze(val_stds[:, slice_id])
                z_annot = str(np.around(slices_z_locs[slice_id], 4))
                title = "slice {:d} @ z = {} um, t = {} um"
                title = title.format(slice_id + 1, z_annot, thickness_annot)

                # Measurmements
                fname = utils.make_fname(fpath, "_" + str(np.int(slice_id + 1)))
                surface.plot_slice(X, Y, slice, label,
                                title = title, fname = fname,
                                cbar_lims = cbar_lims, center = center,
                                n_levels = n_levels, z_aux = Z_aux)
                # STDS
                fname = utils.make_fname(fpath_stds, "_" + str(np.int(slice_id + 1)))
                surface.plot_slice(X, Y, slice_stds, "$\sigma_{norm}$",
                                title = title, fname = fname,
                                cbar_lims = cbar_lims_stds, center = False,
                                z_aux = Z_aux)

            # Plot also Surface measuremnets
            slice = np.squeeze(v_surfs[:, i])
            title = "slice {:d} @ {}".format(0, "surface")
            fname = utils.make_fname(fpath, "_" + "0")
            surface.plot_slice(X, Y, slice, label,
                            title = title, fname = fname, center = center,
                            cbar_lims = (np.nanmin(slice), np.nanmax(slice)),
                            n_levels = n_levels, z_aux = Z_aux)

            plt.close('all')

            # Save parameters
            fname_params = utils.make_fname(   fpath_, subdirname = "slices",
                                                suffix = "_params", ext = ".json")
            param_dict = {  "thickness": thickness,
                            "n_levels": n_levels,
                            "center": center,
                            "z_range": zrange,
                            "clip": clip,
                            "scale": scaleby,
                            "descriptor": descriptor,
                            "overlay": True if Z_aux is not None else False}
            if adjust:
                params_dict.update({"saliency": params_adj})
            utils.save_dict(param_dict, fname_params)

    def plot_surface(   self, plot_current = False, plot_slices = False, n_slices = 10,
                        center = False, thickness = 0.9, zrange = (None, None),
                        clip = (None, None), scaleby = "hop", n_levels = 10,
                        descriptor = "mean", overlay = False, adjust = False):
        """Plot surface as contours and 3D

        Parameters
        -----------
        thickens"""
        # Plot downsampled Data
        result = self.dsdata
        X = np.squeeze(result["X(um)"])
        Y = np.squeeze(result["Y(um)"])

        if self.is_constant_distance:
            # We care about measurements of current
            Z = np.squeeze(result["Current1(A)"])
            z_lab = "Current1(A)"

            ## Pick current values with CONSISTENT COORDINATE ONLY
            ## This works as long as the movemement in Z is just due to noise
            # Z_aux = np.squeeze(result["Z(um)"])

            ## DOWNSAMPLE
            # uniqs, cnts = np.unique(Z_aux, return_counts = True)
            # to_keep = np.nonzero(Z_aux == uniqs[np.argmax(cnts)])[0]
            # X, Y, Z = self._downsample_surface_data(X, Y, Z, to_keep)

            ## AGGREGATE: preferred!
            X, Y, Z = self._aggregate_surface_data(X, Y, Z)

        elif plot_current and not plot_slices:
            # if you want to plot slices, allow leveling Z
            Z = np.squeeze(result["Current1(A)"])
            z_lab = "Current1(A)"
        else:
            Z = np.squeeze(result["Z(um)"])
            z_lab = "Z(um)"

        X, Y, Z = self._remove_last_datapoint(X, Y, Z)
        is_interactive = utils.check_if_interactive()
        # Level Z coordinates

        Z_corr, Z_tilt = analysis.level_plane(  X, Y, Z, True, is_interactive,
                                                z_lab = z_lab)

        if plot_slices:
            Z_aux = Z_corr if overlay else None
            self.plot_slices(   X, Y, Z_tilt, n_slices, thickness, zrange, clip,
                                scaleby, center, n_levels, descriptor, Z_aux,
                                adjust)
        surface.plot_surface_contours( X, Y, Z_corr, z_lab, self.get_fpath(),
                                       center, n_levels)
