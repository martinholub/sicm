import numpy as np
from sicm import io
from os import path

class Experiment(object):
    """Base class for all experiments"""
    def __init__(self, datadir, exp_name):
        self.name = exp_name
        self.datadir = datadir
        data, self.date = self._load_data(datadir, exp_name)
        self.data, self.dsdata, self.idxs = self._downsample_data(data)
        self._data = data

    def _load_data(self, datadir, exp_name):
        # Get files
        files, date = io.get_files(datadir, exp_name)
        result = io.load_result(files, exp_name)
        return result, date

    def _downsample_data(self, data):
        # Select Line Number
        try:
            uniqs, cnts = np.unique(data["LineNumber"], return_counts=True)
            linenos = np.arange(5, max(uniqs), 3)
            result, idxs = io.downsample_to_linenumber(data, linenos)
            dsdata = {k:v[idxs] for k,v in result.items()}
        except Exception as e:
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
