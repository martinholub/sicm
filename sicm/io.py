# Function defs
import numpy as np
import sys
import os
import glob
import csv
import re
import time
import copy
import pandas as pd
from matplotlib.image import imread

def load_result(files = [], exp_name = "", ext = "tsv", sep = "\t"):
    """Load data for files
    """
    result = {}
    for i, f in enumerate(files):
        try:
            name = re.search(exp_name + "_(.*)." + ext, os.path.basename(f)).group(1)
        except AttributeError as e:
            name = exp_name + "_" + str(i)
        name = name.replace(" ", "")
        with open(f, "r") as rf:
            tsvf = csv.reader(rf, delimiter = sep)
            all_rows = []
            for i, row in enumerate(tsvf):
                try:
                    row = np.asarray(list(map(float, filter(None, row))))
                except ValueError as e:
                    import pdb; pdb.set_trace()
                all_rows.append(row)
            if len(all_rows) > 1:
                result[name] = np.asarray(all_rows).reshape((len(all_rows), len(row)))
            else:
                result[name] = all_rows[0]

    print("Avaliable data:"); print(result.keys())
    num_vals = set(x.shape for x in result.values())
    assert len(num_vals) <= 1, "Not all arays are of same length."
    print("Number of datapoints = {}".format(list(num_vals)[0]))
    return(result)

def get_files(datadir, exp_name, ext = "tsv"):
    """Get all files in directory

    Fetches a time of modification of the files.
    """
    all_files = glob.glob(os.path.join(datadir, exp_name + "*." + ext))
    # Discard files with similar basename
    if len(all_files) > 1:
        r = re.compile(exp_name + "_(.*)." + ext)
        files = [f for f in all_files if r.search(f)]
    else:
        files = all_files
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
    idx = np.argwhere(tf) # Select all with this line number
    # Return also index to raw data, if needed
    idxs_ = []
    # Only last or fist element point of approach
    if which == "last":
        ret_idx = np.argwhere(tf_diff == -1)
        for dl in lineno:
            idxs_.append(np.max(np.argwhere(result["LineNumber"] == dl)))
    elif which == "first":
        ret_idx = np.argwhere(tf_diff == 1)
        for dl in lineno:
            idxs_.append(np.min(np.argwhere(result["LineNumber"] == dl)))
    else:
        ret_idx = idx # Select all with this line number
        idxs_ = np.argwhere(np.isin(result["LineNumber"], lineno))
    # Get indices to the sub'idx'ed array
    ret_idx = np.nonzero(np.in1d(idx, ret_idx))[0]
    if not isinstance(idxs_, (np.ndarray)): idxs_ = np.asarray(idxs_)

    for k,v in result.items():
        if k == "dt(s)":
            result_out["time(s)"] = np.cumsum(v)[idx]
        result_out[k] = v[idx]

    print("Number of datapoints = {}".format(len(ret_idx)))

    return(result_out, ret_idx, idxs_)

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
        bad_rows = set(["count", "nexttimestamp", "settimestamp"])
        for i, row in enumerate(ssvf):
            if int(row[0]) != chunk: continue
            if not first_line: # collect information shared by all variables in chunk
                first_line = i
                timestamp = int(row[1])
                size = int(row[2])
            if row[3] in bad_rows: continue
            filt_row = filter(lambda x: x != "nan", filter(None, row[4:]))
            result[row[3]] =  np.asarray(list(map(float, filt_row)))

    if len(set(map(len, result.values()))) not in (0, 1):
        raise ValueErrorr('not all arrays have same length!')

    # Their timestamp is not what we expect, take it from file modification time
    timestamp = os.path.getmtime(fname)
    t = time.localtime(timestamp)
    date = "{:02d}/{:02d}/{:04d} {:02d}:{:02d}".format(t.tm_mday, t.tm_mon, t.tm_year,
                                                       t.tm_hour, t.tm_min)

    print("Experiment time: {}, # of points: {}".format(date, size))

    return(result, date)

def load_comsol(fname):
    """Load table data generated by COMSOL"""
    # Get date of file
    date = os.path.getmtime(fname)
    t = time.localtime(date)
    date = "{:02d}/{:02d}/{:04d} {:02d}:{:02d}".format(t.tm_mday, t.tm_mon, t.tm_year,
                                                       t.tm_hour, t.tm_min)
    # Read file contents
    with open(fname, "r") as rf:
        X = []
        Y = []
        n_dpoints = 0

        for line in rf:
            if line.startswith("%"):
                line = re.sub("^%\s", '', line)
                if line.startswith("Date"):
                    date_ = re.match("Date:\s+(.*)$", line).group(0)
                headline = line
                continue
            if n_dpoints == 0:
                rsub = re.compile("\s{3,}") # replace repeated spaces with tabs
                headline = list(filter(None, re.sub(rsub, "\t", headline).split("\t")))
                # extremely dirty way how to deal with the fact that comsol has
                # variable field separator.
                unit = headline[-1][-4:-1]
                l_end = headline.pop()
                headline.extend([l + unit for l in l_end.split(unit)[:-1]])
            vals = list(filter(None, re.sub("\s+", "\t", line).split("\t")))
            x = np.float32(vals[0])
            y = tuple(map(np.float32, tuple(vals[1:])))
            X.append(x)
            Y.append(y)
            n_dpoints += 1

        X = np.asarray(X)
        Y = np.squeeze(np.asarray(Y).reshape((len(X), len(y))))
        try:
            XY = np.stack((X, Y))
        except ValueError as e: # all input arrays must have the same shape
            # occirs when mopre than one y
            XY = np.concatenate((X[:, np.newaxis], Y), axis = 1).T
        ret = {k: v for k,v in zip(headline, XY)}
        pd_ret = pd.DataFrame(ret)

        return pd_ret, date

def _combine_paths(fileparts):
    """Combines parts of filename to a single path

    Parameters
    -----------
    fileparts: array-like
        Parts of path to be joined with sys separator.

    """
    if isinstance(fileparts, (list, tuple)):
        if len(fileparts) > 1:
            fname = os.path.join(fileparts[0], *fileparts[1:])
        else:
            fname = fileparts[0]
    else:
        fname = fileparts
    return fname

def save_dataframe(df, fileparts):
    """Helper function to pickle dataframe"""
    fname = _combine_paths(fileparts)

    root, ext = os.path.splitext(fname)
    if not ext: # by default save to pickle format
        ext = ".pkl"
    fname = root + ext

    df.to_pickle(fname)
    print("Saved dataframe to {}.".format(fname))

def load_dataframe(fileparts):
    """Load a dataframe

    Parameters
    -----------
    fileparts: array-like
        Parts of path to be joined with sys separator."""
    fname = _combine_paths(fileparts)
    df = pd.read_pickle(fname)
    return(df)

def quoted_data_reader(tsvfile):
    """Generator giving rows in TSV file
    """
    with open(tsvfile, encoding = "ISO-8859-1") as tf:
        reader = csv.reader(tf, dialect="excel-tab")
        for row in reader:
            ## this would work if all rows had same length -.-
            # nested_row = (x.split("\t") for x in row)
            # yield (y for sublist in nested_row for y in sublist)
            yield row

def load_image(imdir, imname):
    """Load image to an array"""
    impath = os.path.join(imdir, imname)
    img = imread(impath)
    return img

def rgb2gray(rgb):
    """Convert RGB to grayscale using standard formula

    References:
    [1]  https://pillow.readthedocs.io/en/3.2.x/reference/Image.html#PIL.Image.Image.convert
    """
    return np.dot(rgb[:,:, :3], [0.2989, 0.5870, 0.1140])

def write_tsv(datadict, dirpath, exp_name, ext = ".tsv", sep = "\t"):
    """Save data to dictionary
    """
    dirpath += "\\" + exp_name
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

    for k,v in datadict.items():
        fname = exp_name + "_{}".format(k)
        this_path = dirpath + "\\" + fname + ext
        with open(this_path, "w", newline = "") as tsvfile:
            writer = csv.writer(tsvfile, delimiter=sep)
            writer.writerow(v)
