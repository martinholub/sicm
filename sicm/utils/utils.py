import time
import matplotlib.pyplot as plt

def save_fig(fname):
    """Helper to save figures"""
    fname = fname.replace(" ", "_").replace("/", "").replace(":", "")
    fname = fname + ".pdf"
    plt.savefig(fname, dpi = 300, bbox_inches = "tight")
