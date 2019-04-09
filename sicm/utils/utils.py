import time
import matplotlib.pyplot as plt
import pandas as pd
import os


def make_fname(fname, suffix = "", ext = ".pdf"):
    basename = os.path.splitext(os.path.basename(fname))[0] + suffix
    dirname = os.path.dirname(fname)
    basename = basename.replace(" ", "_").replace("/", "").replace(":", "").replace(".", "")
    fname = os.path.join(dirname, basename + ext)
    return fname

def save_fig(fname, suffix = "", ext = ".pdf"):
    """Helper to save figures"""
    fname = make_fname(fname, suffix, ext)
    plt.savefig(fname, dpi = 300, bbox_inches = "tight")
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
