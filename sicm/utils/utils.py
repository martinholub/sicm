import time
import matplotlib.pyplot as plt
import pandas as pd
import os

def save_fig(fname):
    """Helper to save figures"""
    fname = fname.replace(" ", "_").replace("/", "").replace(":", "").replace(".", "")
    fname = fname + ".pdf"
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
