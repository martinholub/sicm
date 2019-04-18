import numpy as np
from sicm import io
from os import path
from copy import deepcopy

class Experiment(object):
    """Base class for all experiments

    Parameters
    -------------
    datadir: str
        Directory where data for given experiment is stored
    exp_name: str
        Name of the experiment, must correspond to the filename
    etype: str
        Type of experiment, either "scan" or "it" or "constant_distance" or "approach"
    """
    def __init__(self, datadir, exp_name, etype = "scan"):
        self.name = exp_name
        self.datadir = datadir
        self._assign_exp_parameters(etype)
        data, self.date = self._load_data(datadir, exp_name)
        self.data, self.dsdata, self.idxs = self._downsample_data(data)
        self._data = data

    def _assign_exp_parameters(self, etype):
        if etype.lower() == "constant_distance":
            self.is_constant_distance = True
        else:
            self.is_constant_distance = False

        if etype.lower() == "it":
            self.is_it = True
        else:
            self.is_it = False

        if etype.lower() == "approach":
            self.is_approach = True
        else:
            self.is_approach = False

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
                linenos = np.arange(10, max(uniqs), 6)
                ## if NO_REVERSE, apply heuristics to select the correspnding linenumbers
                # guess_lineno_cnts = np.sort(cnts)[-3]
                # linenos = uniqs[np.argwhere(np.logical_and( cnts > guess_lineno_cnts*0.9,
                #                                             cnts < guess_lineno_cnts*1.1))]
                # result, idxs = io.downsample_to_linenumber(data, linenos, "all")
            elif self.is_it:
                linenos = np.arange(min(uniqs)+1, max(uniqs), 1)
                result, idxs = io.downsample_to_linenumber(data, linenos, "all")
            else:
                # hopping_scan
                linenos = np.arange(5, max(uniqs), 3)
                result, idxs = io.downsample_to_linenumber(data, linenos, which)

            dsdata = {k:v[idxs] for k,v in result.items()}
        except Exception as e:
            if not self.is_approach:
                # For approach, you don'w want to select linenumbers anyway
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

class ExperimentList(object):
    """Class Holding Multiple Experiment Objects"""
    def __init__(self):
        pass
        # self.list = self._create_experiments_list(datadirs, exp_names, etype)

    def _create_experiments_list(self, datadirs, exp_names, objs):
        """ Build List of Experiments

        Parameters
        --------------
        objs: Objects
            List of (deepcopies of) objects of the corresponding class

        Returns
        ----------
        elist: list
            List of instatiated objects of the requred class
        """
        # Make basic checks
        assert isinstance(datadirs, (list, tuple, np.ndarray))
        assert isinstance(exp_names, (list, tuple, np.ndarray))
        assert isinstance(objs, (list, tuple, np.ndarray))

        # Fill parents if any missing
        if len(datadirs) < len(exp_names):
            datadirs = [datadirs[0]] * len(exp_names)
        if len(objs) < len(objs):
            objs = [deepcopy(objs[0]) for i in range(len(exp_names))]

        elist = []
        keys = []
        for dd, en, ob in zip(datadirs, exp_names, objs):
            import pdb; pdb.set_trace()
            experiment = ob(dd, en)
            keys.append(sorted(list(experiment.data.keys())))
            elist.append(experiment)
        test_keys = keys[0]
        assert all(k == test_keys for k in keys[1:])

        # TODO: Can go drop all keys that are not common to all datasets
        return elist
