# Function defs
import numpy as np
import sys
import os
import glob
import csv
import re
import time
import copy

def load_result(files = [], exp_name = ""):
    """Load data for files
    """
    result = {}
    for f in files:
        name = re.search(exp_name + "_(.*).tsv", os.path.basename(f)).group(1)
        name = name.replace(" ", "")
        with open(f, "r") as rf:
            tsvf = csv.reader(rf, delimiter = "\t")
            for i, row in enumerate(tsvf):
                try:
                    row = np.asarray(list(map(float, filter(None, row))))
                except ValueError as e:
                    import pdb; pdb.set_trace()
                result[name] = row
    print(result.keys())
    num_vals = set(x.shape for x in result.values())
    assert(len(num_vals) <= 1)
    print("Number of datapoints = {}".format(list(num_vals)[0]))
    return(result)

def get_files(datadir, exp_name):
    """Get all files in directory

    Fetches a time of modification of the files.
    """
    files = glob.glob(os.path.join(datadir, exp_name + "*.tsv"))
    dates = [os.path.getmtime(f) for f in files]
    t = time.localtime(min(dates))
    date = "{:02d}/{:02d}/{:04d} {:02d}:{:02d}".format(t.tm_mday, t.tm_mon, t.tm_year,
                                                       t.tm_hour, t.tm_min)
    return (files, date)

#Function Defs
def downsample_to_linenumber(result = {}, lineno = -1, which = "last"):
    """Downsample data to select line number

    Parameters
    ---------
    result: dict
        data as k,v pairs
    lineno: int, array-like
        Line Numbers to extract from data, can be multiple [default all]
    which: str
        Either 'last', 'first' or 'all'. Which occurence of line number to get?

    Returns
    ----------
    result_out: dict
        data, subsampled, with new key 'time(s)' added.
    """

    result_out = copy.deepcopy(result)
    if not isinstance(lineno, (list, tuple, np.ndarray)):
        lineno = [lineno]
    if lineno[0] < 0:
        return result_out

    assert "LineNumber" in result.keys()
    assert "dt(s)" in result.keys()

    tf = np.isin(result["LineNumber"], lineno)
    tf_diff = np.diff(tf.astype(np.int))

    if which == "last":
        idx = np.argwhere(tf_diff == -1)
    elif which == "first":
        idx = np.argwhere(tf_diff == 1)
    else:
        idx = np.argwhere(tf)

    for k,v in result.items():
        if k == "dt(s)":
            result_out["time(s)"] = np.cumsum(v)[idx]
        else:
            result_out[k] = v[idx]

    print("Number of datapoints = {}".format(len(idx)))

    return(result_out)


def load_data_lockin(folder = ".", fname = "", chunk = 0):
    """Load data exported by ZHInst LabOne

    Parameters
    -----------
    folder: str
        location of data files
    fname: str
        name of file to load; semicolon delimtied
    chunk: int
        Number of sweep to pull out

    Returns
    -------
    result: dict
        Data values in chunk as key,value pair
    date: str
        Date of last modification of the loaded file
    """
    fname = os.path.abspath(os.path.join(folder, fname))
    assert os.path.isfile(fname), "File does not exist"
    assert isinstance(chunk, (int, )), "Chunk is not an integer"
    print("Exctracting chunk {} from file {}.".format(chunk, fname))

    with open(fname, "r") as rf:
        ssvf = csv.reader(rf, delimiter = ";")
        headline = next(ssvf) # skip header line
        first_line = None
        result = {}
        for i, row in enumerate(ssvf):
            if int(row[0]) != chunk: continue
            if not first_line: # collect information shared by all variables in chunk
                first_line = i
                timestamp = int(row[1])
                size = int(row[2])
            result[row[3]] =  np.asarray(list(map(float, filter(None, row[4:]))))

    if len(set(map(len, result.values()))) not in (0, 1):
        raise ValueErrorr('not all arrays have same length!')

    # Their timestamp is not what we expect, take it from file modification time
    timestamp = os.path.getmtime(fname)
    t = time.localtime(timestamp)
    date = "{:02d}/{:02d}/{:04d} {:02d}:{:02d}".format(t.tm_mday, t.tm_mon, t.tm_year,
                                                       t.tm_hour, t.tm_min)

    print("Experiment time: {}, # of points: {}".format(date, size))

    return(result, date)
