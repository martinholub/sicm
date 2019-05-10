import matplotlib.pyplot as plt
import re
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from sicm.io import quoted_data_reader
from sicm.plots import plots
from sicm.utils import utils

class ZetaSizerReader(object):
    """Class for reading data expored from ZetaSizer Instrument
    """
    def __init__(self, fpath):
        self.fpath = fpath # TODO: check that file exists, make it to list

    def _load_measurement(self, fpath):
        """Load data from tab-separated file to OrderedDictionary

        It is assumed that fileds in the file are quoted and of variable lenght,
        which results in slightly cumbersome procedure.

        TODO: Make the function accept both quoted and unquoted fields.
        """

        # Load rows to dataframe, use column names
        df_ = pd.DataFrame(quoted_data_reader(self.fpath))
        df = pd.DataFrame(df_.values[1:], columns=df_.iloc[0])

        # Unique Samples in the file (includes 3 repeats per sample)
        samples = df["Sample Name"].values.tolist()
        # Unique Datasets in the file (ordered)
        columns = list(re.match("(.*?)(\[[0-9]+\]| ?)( ?\(.*\)| ?)$", x).group(1) for x in
                                df.columns.to_list())
        units = list(re.match("(.*?)(\[[0-9]+\]| ?)( ?)(\(.*\)| ?)$", x).group(4) for x in
                                df.columns.to_list())
        _, idx = np.unique(columns, return_index=True)
        columns = np.asarray(columns)[np.sort(idx)].tolist()
        units = np.asarray(units)[np.sort(idx)].tolist()

        # Populate dictionary of the form {name: {repeat: {col: values}}}
        vals = OrderedDict()
        for sample in samples:
            # Get basename and repeat index of current sample
            re_match = re.match("(.*) ([1-9])$", sample)
            name = re_match.group(1)
            repeat = re_match.group(2)
            # Initialize entry in dictionary
            if name not in vals.keys():
                vals[name] = {}
            # pull out row
            row = df[df["Sample Name"] == sample].values.tolist()[0]
            for i, (col, unit) in enumerate(zip(columns, units)):
                # Figure out if field contains an array
                this_field = row[i].split("\t")
                if len(this_field) == 1:
                    this_field = this_field[0]
                else:
                    this_field = np.asarray(this_field, dtype = np.float)
                if col in vals[name].keys():
                    vals[name][col]["value"].append(this_field)
                else:
                    vals[name].update({col: {"value": [this_field], "unit": unit}})

        uniq_samples = list(vals.keys())
        objs = []
        for uniq_sample in uniq_samples:
            objs.append(DLS(vals[uniq_sample], uniq_sample, self.fpath))
        return objs

    def load_measurements(self):
        fpaths = self.fpath
        objs = []
        if not isinstance(fpaths, (list, tuple, np.ndarray)): fpaths = [fpaths]
        for fpath in fpaths:
            obj = self._load_measurement(fpath)
            objs.extend(obj)
        self.measurements = objs
        return objs


class DLS(object):
    """Dynamic Light Scattering Measurmement
    """
    def __init__(self, dset, name, fpath):
        self.data = dset
        self.fpath = fpath
        self.name = name
        self._describe_dataset()

    def _describe_dataset(self):
        """Pull out annotation information from dataset

        This assumes a convention for exporting data.
        """
        annot = {}
        annot["type"] = self.data["Type"]["value"][0]
        annot["samples"] = self.data["Sample Name"]["value"]
        annot["date"] = self.data["Measurement Date and Time"]["value"][-1]
        try:
            annot["peaks"]={"1":(np.mean(np.asarray(self.data["%Pd Peak 1"]["value"], np.float)),
                                np.std(np.asarray(self.data["%Pd Peak 1"]["value"], np.float))),
                            "2": (np.mean(np.asarray(self.data["%Pd Peak 2"]["value"], np.float)),
                                np.std(np.asarray(self.data["%Pd Peak 2"]["value"], np.float))),
                            "3": (np.mean(np.asarray(self.data["%Pd Peak 3"]["value"], np.float)),
                                np.std(np.asarray(self.data["%Pd Peak 3"]["value"], np.float)))}
        except Exception as e:
            pass

        annots = [  ("z-average [nm]", "Z-Average"),
                    ("pdi", "PdI"),
                    ("pdi-width", "PdI Width"),
                    ("snr", "Signal To Noise Ratio"),
                    ("D [um2/s]", "Diffusion Coefficient"),
                    ("molecular-weight [kDa]", "Molecular Weight"),
                    ("volume-mean", "Volume Mean"),
                    ("intensity-mean", "Intensity Mean"),
                    ("number-mean", "Number Mean")]
        for (t,s) in annots:
            try:
                annot[t]=(np.mean(np.asarray(self.data[s]["value"], np.float)),
                        np.std(np.asarray(self.data[s]["value"], np.float)))
            except Exception as e:
                pass

        self.annotation = annot

    def plot(self, ykey, xkey = "Sizes"):
        """Plot measurmement for a sample

        Parameters
        ---------------
        xkey: str
            Key indicating which values to plot on x-axis.
        ykey: str
            Key indicating which values to ploton y-axis.
        """
        y = np.mean(self.data[ykey]["value"], axis = 0)
        y_err = np.std(self.data[ykey]["value"], axis = 0)
        y_lab = "{} {}".format(ykey, self.data[ykey]["unit"])

        x = np.mean(self.data[xkey]["value"], axis = 0)
        x_err = np.std(self.data[xkey]["value"], axis = 0)
        x_lab = "{} {}".format(xkey, self.data[xkey]["unit"])

        if any(x_err/x > 0.1 * x):
            raise Exception("Coordinates on X-axis differ among samples!")

        legend = self.name.replace("_sio2SOP", "")
        fname = os.path.splitext(self.fpath)[0] + "_" + self.name + "_" + ykey
        plots.errorplot_generic([x], [y] , [y_err], x_lab, y_lab, legend, fname)
