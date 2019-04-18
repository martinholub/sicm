
import numpy as np
import os
from matplotlib import pyplot as plt
from copy import deepcopy

from sicm import plots
from sicm.utils import utils
from .experiment import Experiment, ExperimentList
from ..measurements.signal import Signal

class Approach(Experiment):
    def __init__(self, datadir, exp_name):
        super(Approach, self).__init__(datadir, exp_name, etype = "approach")

    def plot(self, sel = None, what = "generic"):
        """Simple wrapper for plotting
        """
        if what.lower().startswith("sicm"):
            plots.plot_sicm(self.dsdata, sel, "Approach", self.name, self.date)
        else:
            # consider calling plot_generic directly
            z_ax = self.dsdata["Z(um)"]
            z_ax = z_ax - np.min(z_ax)
            y_ax = self.dsdata["Current1(A)"]*1e9
            if sel is None:
                sel = [True] * len(z_ax)

            sig = Signal(z_ax[sel], y_ax[sel], self.datadir, self.name)
            sig.plot("Z [um]", "Current [nA]", legend = "")


class ApproachList(ExperimentList):
    """Construct List of Approach Objects
    """
    def __init__(self, datadirs, exp_names):
        self.list = self._create_approaches_list(datadirs, exp_names)

    def _create_approaches_list(self, datadirs, exp_names):
        """ Build List of Approaches """
        objs = [deepcopy(Approach) for i in range(len(exp_names))]
        alist = self._create_experiments_list(datadirs, exp_names, objs)
        return alist

    def _stitch(self, datasets_str):
        """Stitch datasest for elements of lists

        Parameters
        ---------------
        datasets_str: array-like
            List of attributes' (datasets') names to stitch

        Returns
        -----------
        stitch_obj: Approach
            object of Approach class, stitched approach
        stitch_lengths: array-like
            lengths of individual objets that were stitched
        """

        def _stitch_dicts(dict1, dict2):
            """Helper fun
            Combine keys for dictinaries with identical keys"""
            # Pull out keys and values
            keys1 = dict1.keys()
            keys2 = dict2.keys()
            values1 = dict1.values()
            values2 = dict2.values()
            # Dont attempt anythong fancy and fail if keys are not identical
            assert keys1 == keys2

            # Stitch values
            try: # Assume arrays
                stitch_dict = {k1: np.concatenate((v1, v2)) for (k1, v1, v2) in
                                zip(keys1, values1, values2)}
            except Exception as e: # some generic exception
                try: # Assume lists and join them
                    stitch_dict = {k1: v1 + v2 for (k1, v1, v2) in
                                    zip(keys1, values1, values2)}
                except Exception as e: # some generic exception
                    raise e # give up
            return stitch_dict

        # Stitch by appending to the first element
        stitch_obj = deepcopy(self.list[0])
        stitch_lengths = [len(getattr(o, datasets_str[0])["time(s)"]) for o in self.list]

        # Stitch only selected attributes of objects in list
        for dset_str in datasets_str:
            for i in range(len(self.list) - 1):
                if i == 0:
                    dset1 = deepcopy(getattr(stitch_obj, dset_str))
                else:
                    dset1 = dset_stitch
                dset2 = deepcopy(getattr(self.list[i + 1], dset_str))
                try:
                    dset_stitch = _stitch_dicts(dset1, dset2)
                except Exception as e:
                    raise e

            # update attribute of base stitch_obj my combined dataset
            setattr(stitch_obj, dset_str, dset_stitch)

        # Make stich name a list
        # TODO: When other functions fail, do:
        # stitch_name = ",".join((obj.name for obj in self.list))
        stitch_obj.name = [obj.name for obj in self.list]

        return stitch_obj, stitch_lengths

    def _overlap(   self, obj, lengths, datasets, z_move = 25, z_range = 30,
                    preserve_overlap = False):
        """Overlaps selected datasets of previously stitched objects

        Parameters
        ---------------
        obj: Approach
            Stitched Approach
        lengths: array-like
            lengths of individual segments of stittched object
        datasets: array-like
            Lists of strings, names of attributes of obj to be stitched
        z_move: float
            z-shift between segments, in um
        z_range: float
            maximal z-range of each segment, needed only if preserve_overlap == False
        preserve_overlap: bool
            Preserve overlapping segments in stitched datasets?

        Returns
        -----------
        obj_out: Approach
            Overlapped Approach
        """
        # General value of overlap between nano and microdrive movements
        z_overlap = z_range - z_move

        # Make deepcopy of the object that will be adjusted
        obj_out = deepcopy(obj)
        for dset_name in datasets: # for each selected dataset
            dset = getattr(obj, dset_name)
            # Get z-values and obtain copy
            z = dset["Z(um)"]
            z_ = deepcopy(z)
            # Select only non-verlapping z-coordinates, if requested
            if not preserve_overlap:
                keep_id = z > z_overlap # z goes from z-range to 0!
            else:
                keep_id = [True] * len(z)

            # Loop over parts of stitched z-axis and add known cummulative shift
            this_id = 0
            for i, id in enumerate(lengths):
                if i >= len(lengths) - 1:
                    continue
                this_id += id
                z_[0:this_id] += z_move

            # Drop overlapping z-coordinates if requested
            z_ = z_[keep_id]
            # Get dt value and make sure it is the correct one
            uniqs, cnts = np.unique(dset["dt(s)"], return_counts=True)
            dt = uniqs[np.argmax(cnts)]
            # Construct time-axis
            t_ = np.cumsum([dt]*len(z_))

            # Reassing adjusted key:value pairs
            dset_ = deepcopy(dset)
            for i, k in enumerate(dset.keys()):
                if k == "time(s)":
                    dset_[k] = t_
                elif k == "Z(um)":
                    dset_[k] = z_
                else:
                    dset_[k] = dset[k][keep_id]

            # Make sure changes contained in this function, assign and return
            setattr(obj_out, dset_name, dset_)
        return obj_out


    def stitch(self, z_move = 25, z_range = 30, preserve_overlap = True):
        """Stitch multiple Approaches to single dataset

        Parameters
        ---------
        overlap: float
            Fractional size of overlap between
        """
        # Decide which datasets to stitch & overlap
        # Currently this ignores IDX, but as we tak all idx, for each segment, this is OK.
        datasets = ["data", "_data", "dsdata"]
        stitch_obj, stitch_lengths = self._stitch(datasets)
        overlap_obj = self._overlap(stitch_obj, stitch_lengths, datasets,
                                    z_move, z_range, preserve_overlap)
        self.stitched = overlap_obj
        return overlap_obj
