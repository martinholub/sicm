import numpy as np
import os
from matplotlib import pyplot as plt
from copy import deepcopy

from sicm import analysis, io
from sicm.utils import utils
from .signal import Signal
from sicm.plots import plots

class Hops(object):
    def __init__(self, data, idxs, name, date):
        self.data = data
        self.idxs = idxs
        self.name = name
        self.date = date

    def _select_data(self, sel = None):
        """Make selection of data to plot.

        Parameters
        -----------
        sel: array-like
            Either (A) indices to data to select, or (B) array-like [<id1>, <id2>]
            both <= 1.0 or (C) array-like [<time1>, <time2>] indicating time range.

        Returns
        ---------------
        subresult: dict
            Data with keys limited to `subkeys` and values selected with `sel`
        idxs: array-like
            Presumed location of extrema
        """
        if sel is None: # take all
            sel = np.arange(0, len(next(iter(self.data.values()))))

        if all(np.asarray(sel) <= 1): # assme fractional indexing
            assert len(sel) == 2
            max_len = len(next(iter(self.data.values())))
            sel = np.arange(int(sel[0]*max_len), int(sel[-1]*max_len))
        elif isinstance(sel[0], (float, )) and isinstance(sel[-1], (float)):
            # assume time indexing
            assert len(sel) == 2
            X = np.squeeze(self.data["time(s)"])
            sel = np.nonzero(np.logical_and(X > sel[0], X <= sel[-1]))[0]

        subkeys = ['Z(um)', 'LockinPhase', 'Current1(A)', 'time(s)']
        # subkeys.extend(["LockinAmplitude"]) # comment if not desired
        subresult = {k:v[sel] for k,v in self.data.items() if k in subkeys}
        idxs = self.idxs[np.isin(self.idxs, sel)]
        idxs = idxs - sel[0]
        return subresult, idxs

    def plot(self, sel = None, xkey = "time(s)", fname = None, do_annotate = True):
        """Plots all keys in data against x-key

        As plot lockin but on a single plot. Bit of code duplication, but gives
        flexibility later.

        References
        -----------
        [1]: https://matplotlib.org/examples/axes_grid/demo_parasite_axes2.html
        [2]: https://matplotlib.org/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
        """
        data, guessid = self._select_data(sel)
        name = self.name
        date = self.date

        assert isinstance(xkey, (str, )), "xkey must be a string"
        assert xkey in data.keys(), "xkey must be in data"

        # Create single plot with three axis
        # plt.style.use("seaborn")
        plots._set_rcparams()
        fig, ax = plt.subplots(figsize = (4.8/0.75, 4.8))
        plt.subplots_adjust(right = 0.75)
        axs = [ax] + list(map(lambda x: x.twinx(), [ax]*(len(data) - 2)))

        # Adjust the right-most axis
        if len(axs)>3:
            axs[-1].spines["right"].set_position(("axes", 1.2))
            axs[-1].get_yaxis().get_offset_text().set_x(1.3)
            utils.make_patch_spines_invisible(axs[-1])
            axs[-1].spines["right"].set_visible(True)

        # Add name to figure
        if name is None:
            text = r"I-$\theta$-z relation"
        else:
            text = os.path.splitext(name)[0]
        if date:
            text = "{} @ {}".format(text, date)
        fig.suptitle(text, size = 16, y = 0.96)

        fmts_map = {
            'LockinPhase': "-gray",
            'Z(um)': "-k",
            "Current1(A)": "-green"
        }
        handles = []
        labels = []
        if do_annotate:
            data_, annot = self.annotate_peaks(sel, xkey, do_plot = False)
        else:
            data_ = deepcopy(data)

        for i, (k, v) in enumerate(data_.items()):
            if k == xkey: continue # dont plot x vs x
            fmt = fmts_map[k]
            try:
                this_color = fmt[1:]
                this_style = fmt[0]
                axs[i].plot(data_[xkey], v, ls = this_style, c = this_color,
                            label = k, alpha = .5)

                if do_annotate:
                    peaks_id = annot[k.lstrip("_")]["peaks_id"]
                    axs[i].plot(data_[xkey][peaks_id], v[peaks_id], alpha = 1,
                                linestyle = "", marker = "*", markersize = 10,
                                markerfacecolor = this_color)

                if i == 0: axs[i].set_xlabel(xkey)
                axs[i].set_ylabel(k, color = this_color)
                axs[i].tick_params("y", colors = this_color)
                axs[i].grid(axis = "y", color = this_color,
                            alpha = .3, linewidth = .5, linestyle = ":")
                h, l = axs[i].get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
                # axs[i].set_title(" ".join(labels[k[1]].split(" ")[:-1]))
            except KeyError as e:
                plot_mock(axs[i])
        # Combine legends and show.
        ax.legend(handles, labels, bbox_to_anchor = (1., 1.), frameon = True)

        if fname is not None:
            utils.save_fig(fname)
        plt.show()


    def annotate_peaks( self, sel = None, xkey = "time(s)", window_size = 50,
                        do_plot = False, save_dir = None):
        """Get information on peaks

        Parameters
        ----------
        data: dict
        xkey: str
        guessid: array-like
            Indices at or around which peaks are expected
        window_size: int
            Half-size of window to inspect for extrema around guessid
        do_plot: bool

        Returns
        -------------
        data_: dict
            Data as before, except for phase that was detrended.
            TODO: Later can consdier detrending also other keys.
        annot: pandas.DataFrame
            Structrue with information on peaks obtained for each array in data, except for
            the one stored under xkey.
        """
        data, guessid = self._select_data(sel)
        data_ = deepcopy(data)
        annot = {}
        for i, (k, v) in enumerate(data.items()):
            if k == xkey: continue # dont plot x vs x
            if "phase" in k.lower():
                sig = Signal(x = data[xkey].flatten(), y = v.flatten())
                if sel is None:
                    v = sig.detrend_signal(do_plot)
                data_[k] = v # assing detrendend phase
            peaks_id, peaks_annot = analysis._find_peaks(v, guessid, window_size)
            annot.update({k: {"peaks_id": peaks_id,
                            "peaks_val": v[peaks_id].flatten(),
                            "peaks_times": data[xkey][peaks_id].flatten(),
                            "baseline": np.asarray([a[0] for a in peaks_annot]).flatten(),
                            "rel_change": np.asarray([a[1] for a in peaks_annot]).flatten()
                            }})
        annot = utils.annotate_dframe(annot)
        if save_dir is not None:
            io.save_dataframe(annot, [save_dir, self.name + "_annot"])
            if do_plot:
                self.plot_annotation(annot)
        return data_, annot

    def plot_annotation(self, annot, key = "Current1(A)"):
        plt.style.use("seaborn")

        fig, axs = plt.subplots(nrows = 4, ncols = 1, sharex = True)
        axs = axs.flatten()
        fig.text(0.5, 0.04, "idx", ha='center')
        fig.suptitle("Peaks' Annotation", size  = 16, y = 0.96)

        axs[0].plot(annot[key]["baseline"], '^')
        axs[0].set_ylabel(key)
        axs[1].plot(annot[key]["peaks_val"], 's')
        axs[1].set_ylabel(key)
        axs[2].plot(annot[key]["rel_change"])
        axs[2].set_ylabel("rel_change")
        axs[3].plot(annot[key]["peaks_val"]/annot[key]["baseline"])
        axs[3].set_ylabel("rel_value")
