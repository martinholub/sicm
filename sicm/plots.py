
import numpy as np
import os
from copy import deepcopy
from textwrap import wrap
import itertools

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits import mplot3d

from sicm import analysis
from sicm.utils import utils

plt.style.use("seaborn")

def plot_mock(ax):
    """Render a mock plot"""
    ax.text(x=0.5, y=0.5, s = "No Data Available",
            horizontalalignment = "center", wrap = "True",
            fontsize = "xx-large",
            verticalalignment="center", transform=ax.transAxes)

def plot_hopping_scan(result , sel = None, exp_name = "exp", date = "00/00/0000 00:00"):
    """Plot results of hopping scan

    If data is aquired with QTF setup, voltage and current are not available.
    """
    plt.style.use("seaborn")

    fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (10, 15))
    axs = axs.flatten()
    fig.suptitle("Hopping Scan", size = 20, y = 0.92)

    if sel is None:
        sel = np.arange(0, len(result["LineNumber"]))
    if "time(s)" not in result.keys():
        result["time(s)"] = np.cumsum(result["dt(s)"])

    try:
        axs[0].plot(result["time(s)"][sel], result["V1(V)"][sel])
        axs[0].legend(["{} @ {}".format(exp_name, date)])
        axs[0].set_xlabel("time [s]")
        axs[0].set_ylabel("potential [V]")
    except KeyError as e:
        plot_mock(axs[0])

    try:
        axs[1].plot(result["time(s)"][sel], result["Current1(A)"][sel])
        axs[1].legend(["{} @ {}".format(exp_name, date)])
        axs[1].set_xlabel("time [s]")
        axs[1].set_ylabel("current [A]")
    except KeyError as e:
        plot_mock(axs[1])

    axs[2].plot(result["LineNumber"][sel], result["Z(um)"][sel])
    axs[2].legend(["{} @ {}".format(exp_name, date)])
    axs[2].set_xlabel("LineNumber")
    axs[2].set_ylabel("z [um]")

    axs[3].plot(result["time(s)"][sel], result["Z(um)"][sel])
    axs[3].legend(["{} @ {}".format(exp_name, date)])
    axs[3].set_xlabel("time [s]")
    axs[3].set_ylabel("z [um]")

    axs[4].plot(result["LineNumber"][sel],  result["X(um)"][sel])
    axs[4].legend(["{} @ {}".format(exp_name, date)])
    axs[4].set_xlabel("LineNumber")
    axs[4].set_ylabel("x [um]")

    axs[5].plot(result["LineNumber"][sel],  result["Y(um)"][sel])
    axs[5].legend(["{} @ {}".format(exp_name, date)])
    axs[5].set_xlabel("LineNumber")
    axs[5].set_ylabel("y [um]")

    # Plot also results from lockin
    subkeys =  [("time(s)", "LockinPhase"), ("time(s)", "LockinAmplitude")]
    sel_keys = set([y for x in subkeys for y in x])
    subresult = {k:v[sel] for k,v in result.items() if k in sel_keys}
    plot_lockin(subresult, subkeys, name = exp_name, date = date)

def plot_surface(result):
    """Plot surface as contours and 3D"""

    result = analysis.correct_for_current(result)

    X = np.squeeze(result["X(um)"])
    Y = np.squeeze(result["Y(um)"])
    Z = np.squeeze(result["Z(um)"])

    # Level Z coordinates and convert to matrix with proper ordering
    X_sq, Y_sq, Z_sq = analysis.level_plane(X, Y, Z, True)

    plt.style.use("seaborn")
    fig = plt.figure(figsize = (12, 10))
    fig.tight_layout()

    # Filled countour with triangulation
    ax = fig.add_subplot(2, 2, 1)
    C = ax.tricontourf(X, Y, Z_sq.flatten() , cmap='viridis')
    CB = fig.colorbar(C)
    ax.set_xlabel('X(um)')
    ax.set_ylabel('Y(um)')
    ax.set_title('Z(um)')

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    # Surface in 3D projection
    ax.plot_trisurf(X, Y , Z_sq.flatten(), cmap='viridis')
    ax.set_xlabel('X(um)')
    ax.set_ylabel('Y(um)')
    ax.set_zlabel('Z(um)')
    # Filled contours without triangulation
    ax = fig.add_subplot(2, 2, 3)
    C = ax.contourf(X_sq, Y_sq, Z_sq, cmap='viridis')
    CB = fig.colorbar(C)
    ax.set_xlabel('X(um)')
    ax.set_ylabel('Y(um)')
    ax.set_title('Z(um)')

    plt.show()

def plot_lockin(data = {}, keys = [("frequency", "r")], date = None, name = None,
                xlog = False):
    """Plot data collected by lockin amplifier

    For description of variables and units, see [1].

    Parameters
    -----------
    data: dict
        Key,value pairs of variables' values.
    keys: list of tuples
        Pairs of variables to plot on x,y axes
    date: str
        Date of the experimetn

    References:
      [1] https://www.zhinst.com/sites/default/files/LabOneProgrammingManual_42388.pdf
    """
    if not isinstance(keys, (list, np.ndarray)): keys = [keys]
    nplots = len(keys)
    nrows = nplots // 2
    if nplots % 2 == 1: nrows += 1
    ncols = 2 if nplots > 1 else 1

    # units mapping
    labels = {"frequency": "f [Hz]",
             "r": "amplitude [V]",
             "phase": r'$\theta$ [rad]',
             "phasepwr": r'$\theta^2$ [$rad^2$]',
             "x": "x-value [V]", "y": "y-value [V]",
             "LockinAmplitude": "LockIn amplitude [V]",
             "LockinPhase": r"LockIn $\theta$ [rad]",
             "time(s)": "time [s]"}

    plt.style.use("seaborn")
    fig, axs = plt.subplots(nrows, ncols, squeeze = False,
                            figsize = (ncols*6.4, nrows*4.8))
    axs = axs.flatten()
    #     fig.tight_layout()
    if name is None:
        text = "Oscilator"
    else:
        text = os.path.splitext(name)[0]
    if date:
        text = "{} @ {}".format(text, date)
    fig.suptitle(text, size  = 16, y = 0.96)

    for i, k in enumerate(keys):
        try:
            axs[i].plot(data[k[0]], data[k[1]])
            axs[i].set_xlabel(labels[k[0]])
            axs[i].set_ylabel(labels[k[1]])
            if xlog:
                axs[i].set_xscale("log")
                axs[i].set_xlabel("log " + labels[k[0]])
            # axs[i].set_title(" ".join(labels[k[1]].split(" ")[:-1]))
        except KeyError as e:
            plot_mock(axs[i])
    plt.show()

def plot_approach(  data, xkey, guessid = np.array([]),
                    name = None, date = None):
    """Plots all keys in data against x-key

    As plot lockin but on a single plot. Bit of code duplication, but gives
    flexibility later.

    References
    -----------
    [1]: https://matplotlib.org/examples/axes_grid/demo_parasite_axes2.html
    [2]: https://matplotlib.org/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
    """

    assert isinstance(xkey, (str, )), "xkey must be a string"
    assert xkey in data.keys(), "xkey must be in data"

    # Create single plot with three axis
    plt.style.use("seaborn")
    fig, ax = plt.subplots(figsize = (6.4*1.5, 4.8))
    plt.subplots_adjust(right = 0.75)
    axs = [ax] + list(map(lambda x: x.twinx(), [ax]*(len(data) - 2)))

    # Adjust the right-most axis
    axs[-1].spines["right"].set_position(("axes", 1.2))
    axs[-1].get_yaxis().get_offset_text().set_x(1.3)
    utils.make_patch_spines_invisible(axs[-1])
    axs[-1].spines["right"].set_visible(True)

    # Add name to figure
    if name is None:
        text = r"I-$\theta$-z relation"
    else:
        text = os.path.splitext(name)[0]
    if date:
        text = "{} @ {}".format(text, date)
    fig.suptitle(text, size = 16, y = 0.96)

    fmts = ["k-", "r-", "g-"]
    handles = []
    labels = []
    data_, annot = analysis.annotate_peaks(data, xkey, guessid, do_plot = True)
    for i, (k, v) in enumerate(data_.items()):
        if k == xkey: continue # dont plot x vs x
        peaks_id = annot[k.lstrip("_")]["peaks_id"]
        try:
            axs[i].plot(data_[xkey], v, fmts[i], label = k, alpha = .5)
            this_color = fmts[i][0]
            axs[i].plot(data_[xkey][peaks_id], v[peaks_id], alpha = 1,
                        linestyle = "", marker = "*", markersize = 10,
                        markerfacecolor = this_color)

            if i == 0: axs[i].set_xlabel(xkey)
            axs[i].set_ylabel(k, color = this_color)
            axs[i].tick_params("y", colors = this_color)
            axs[i].grid(axis = "y", color = this_color,
                        alpha = .3, linewidth = .5, linestyle = ":")
            h, l = axs[i].get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
            # axs[i].set_title(" ".join(labels[k[1]].split(" ")[:-1]))
        except KeyError as e:
            plot_mock(axs[i])
    # Combine legends and show.
    ax.legend(handles, labels, bbox_to_anchor = (.65, .2), frameon = True)
    plt.show()

    # sigs = [s for s in data_.keys() if s.lower().startswith(("_", "current"))]
    # analysis.correlate_signals(data_, xkey, sigs)
def plot_generic(Xs, Ys, x_labs, y_labs, legend, fname):
    """Generic ploting function

    This is an attempt of generic function that produces publication quality
    plots. Parameters have their usual meanings.
    """
    # set plotting style
    plt.style.use("seaborn-ticks")
    params = {  "font.family": "serif",
                "font.weight": "normal",
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "xtick.bottom": True,
                "xtick.direction": "in",
                "ytick.left": True,
                "ytick.direction": "in"}
    mpl.rcParams.update(params)
    fmts_prod= itertools.product(["k"], ["-", "--", ":", "-."])
    fmts = ["".join(x) for x in fmts_prod]

    fig = plt.figure(figsize = (4.5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    fmts = fmts[0:len(Xs[0])]
    for x, y, x_lab, y_lab, fmt in zip(Xs, Ys, x_labs, y_labs, fmts):
        line = ax.plot(x, y, fmt)
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)

    if not isinstance(legend, (list, tuple)): legend = [legend]
    legend = ['\n'.join(wrap(l, 20)) for l in legend]
    ax.legend(legend, fontsize = ax.xaxis.label.get_size()-2,
                borderaxespad = 1.1)

    utils.save_fig(fname)
    # recover plotting style
    mpl.rcParams.update(mpl.rcParamsDefault)


def boxplot_generic(x, x_labs = None, y_lab = None, legend = None, fname = None):
    """Generic Boxplot function

    This is an attempt of generic function that produces publication quality
    boxplots. Parameters have their usual meanings.

    References
      [1]: https://stackoverflow.com/a/49689249
    """
    # set plotting style
    plt.style.use("seaborn-ticks")
    params = {  "font.family": "serif",
                "font.weight": "normal",
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "xtick.bottom": False,
                "xtick.direction": "in",
                "ytick.left": True,
                "ytick.direction": "in"}
    mpl.rcParams.update(params)

    if x_labs is None:
        try:
            x_labs = np.arange(1, len(x[0])+1).tolist()
        except ValueError as e:
            x_labs = np.arange(1, len(x)+1).tolist()
    if not isinstance(x_labs, (list, tuple, np.ndarray)): x_labs = [x_labs]

    fig = plt.figure(figsize = (4.5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    bxplt = ax.boxplot(x, labels = x_labs, medianprops = {"color": "black", "linestyle": '-.'})
    ax.set_ylabel(y_lab)

    # legend
    if legend is not None:
        if not isinstance(legend, (list, tuple)): legend = [legend]
        legend = ['\n'.join(wrap(l, 20)) for l in legend]
        # `handletextpad=-2.0, handlelength=0` hides the marker in legend [1]
        ax.legend(legend, fontsize = ax.xaxis.label.get_size()-2,
                    borderaxespad = 1.1,
                    handletextpad=-2.0, handlelength=0)

    if fname is not None:
        utils.save_fig(fname)
    # recover plotting style
    mpl.rcParams.update(mpl.rcParamsDefault)
