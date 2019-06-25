import time
import matplotlib.pyplot as plt
from matplotlib import rcParams as rcparams
import pandas as pd
import os
import re
import json

def make_fname(fname, suffix = "", ext = "", subdirname = ""):
    # Append suffix to filename and remove non-desirable characters
    basename, ext_ = os.path.splitext(os.path.basename(fname))
    basename += suffix
    basename = basename.replace(" ", "_").replace("/", "").replace(":", "").replace(".", "")
    basename = basename.replace("\\", "").replace("@", "_")
    # Obtain absolute location of parent direcotry and normalize
    dirname = os.path.dirname(fname)
    subdir_path = os.path.normpath(os.path.join(dirname, subdirname))
    # Create directory if needed
    if not os.path.isdir(subdir_path):
        os.makedirs(subdir_path)

    if ext:
        ext_ = ext

    if not ext_.replace(".", "").startswith(("png", "pdf", "eps", "svg", "jp", "tif")):
        ext_ = ".svg" #"" # "." + rcparams["savefig.format"]

    fname = os.path.normpath(os.path.join(subdir_path, basename + ext_))
    return fname

def save_fig(fname, suffix = "", ext = ""):
    """Helper to save figures"""
    if ext:
        if not ext.startswith("."):
            ext = "." + ext

    fname = make_fname(fname, suffix, ext)
    plt.savefig(fname, dpi = 600, bbox_inches = "tight")
    print("Saved figure to {}.".format(fname))

def make_patch_spines_invisible(ax):
    """
    Having been created by twinx, ax has its frame off, so the line of its detached spine is invisible.  First, activate the frame but make the patch and spines invisible.
    """
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def annotate_dframe(annot):
    """Convert peak annotation to dataframe"""
    dframe = pd.DataFrame.from_dict(annot, orient = "columns")
    return dframe

def check_if_interactive():
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
            is_interactive = False # jupyter
        else:
            is_interactive = True # ipython
    except NameError as e:
        is_interactive = True # command line
    return is_interactive

def save_dict(d, fname):
    """Save dictionary to JSON
    """
    # Make sure it is json!
    # fname = make_fname(fname, suffix = "", ext = ".json", subdirname = "")
    with open(fname, "w") as wd:
        json.dump(d, wd, indent = 4, sort_keys = True)
def load_dict(fname):
    with open(fname, "r") as rd:
        d = json.load(rd)
    return d
