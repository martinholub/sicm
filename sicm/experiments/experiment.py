import numpy as np
from sicm import io
from os import path

class Experiment(object):
    """Base class for all experiments"""
    def __init__(self, datadir, exp_name, scan_type = "scan"):
        self.name = exp_name
        self.datadir = datadir
        self._assign_scan_parameters(scan_type)
        data, self.date = self._load_data(datadir, exp_name)
        self.data, self.dsdata, self.idxs = self._downsample_data(data)
        self._data = data

    def _assign_scan_parameters(self, scan_type):
        if scan_type.lower() == "constant_distance":
            self.is_constant_distance = True
        else:
            self.is_constant_distance = False

        if scan_type.lower() == "it":
            self.is_it = True
        else:
            self.is_it = False

    def _load_data(self, datadir, exp_name):
        # Get files
        files, date = io.get_files(datadir, exp_name)
        result = io.load_result(files, exp_name)
        return result, date

    def _downsample_data(self, data, which = "last"):
        # Select Line Number
        try:
            uniqs, cnts = np.unique(data["LineNumber"], return_counts=True)
            if self.is_constant_distance:
                # constant_distance scan
                # starting at 6 is forward, starting at 10 is reverse
                linenos = np.arange(6, max(uniqs), 6)
                # if NO_REVERSE, dirty trick for now:
                guess_lineno_cnts = np.sort(cnts)[-3]
                linenos = uniqs[np.argwhere(np.logical_and( cnts > guess_lineno_cnts*0.9,
                                                            cnts < guess_lineno_cnts*1.1))]
                result, idxs = io.downsample_to_linenumber(data, linenos, "all")
            elif self.is_it:
                linenos = np.arange(min(uniqs)+1, max(uniqs), 1)
                result, idxs = io.downsample_to_linenumber(data, linenos, "all")
            else:
                # hopping_scan
                linenos = np.arange(5, max(uniqs), 3)
                result, idxs = io.downsample_to_linenumber(data, linenos, which)

            dsdata = {k:v[idxs] for k,v in result.items()}
        except Exception as e:
            msg = "{} : {}".format(type(e).__name__, e.args)
            print(msg + "\nAll points will be used.")
            idxs = np.arange(0, len(next(iter(data.values()))))
            data["time(s)"] = np.cumsum(data["dt(s)"]) # TODO: Make more robust.
            dsdata = data
            result = data

        return result, dsdata, idxs

    def _get_timestamp(self):
        tstamp = self.date.replace("/", "").replace(":", "").replace(" ", "_")
        return tstamp

    def _get_fname(self):
        fname = "{}_{}".format(self.name, self._get_timestamp())
        return fname

    def get_fpath(self):
        fpath = path.normpath(path.join(self.datadir, self._get_fname()))
        return fpath
