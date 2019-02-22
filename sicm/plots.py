
import numpy as np
import os
from copy import deepcopy
from textwrap import wrap
import itertools
import re

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

def plot_sicm(result, sel, title = "SICM Plot", exp_name = None, date = "00/00/0000"):
    """Generic SICM plot"""

    plt.style.use("seaborn")

    fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (10, 15))
    axs = axs.flatten()
    fig.suptitle(title, size = 20, y = 0.92)

    if sel is None:
        try:
            sel = np.arange(0, len(result["LineNumber"]))
        except KeyError as e:
            sel = np.arange(0, len(next(iter(data.values()))))
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

    try:
        axs[2].plot(result["LineNumber"][sel], result["Z(um)"][sel])
        axs[2].legend(["{} @ {}".format(exp_name, date)])
        axs[2].set_xlabel("LineNumber")
        axs[2].set_ylabel("z [um]")
    except KeyError as e:
        plot_mock(axs[2])

    try:
        axs[3].plot(result["time(s)"][sel], result["Z(um)"][sel])
        axs[3].legend(["{} @ {}".format(exp_name, date)])
        axs[3].set_xlabel("time [s]")
        axs[3].set_ylabel("z [um]")
    except KeyError as e:
        plot_mock(axs[3])

    try:
        axs[4].plot(result["LineNumber"][sel],  result["X(um)"][sel])
        axs[4].legend(["{} @ {}".format(exp_name, date)])
        axs[4].set_xlabel("LineNumber")
        axs[4].set_ylabel("x [um]")
    except KeyError as e:
        plot_mock(axs[4])

    try:
        axs[5].plot(result["LineNumber"][sel],  result["Y(um)"][sel])
        axs[5].legend(["{} @ {}".format(exp_name, date)])
        axs[5].set_xlabel("LineNumber")
        axs[5].set_ylabel("y [um]")
    except KeyError as e:
        plot_mock(axs[5])

    # Plot also results from lockin
    subkeys =  [("time(s)", "LockinPhase"), ("time(s)", "LockinAmplitude")]
    sel_keys = set([y for x in subkeys for y in x])
    subresult = {k:v[sel] for k,v in result.items() if k in sel_keys}
    plot_lockin(subresult, subkeys, name = exp_name, date = date)

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
             "time(s)": "time [s]",
             "grid": "grid"}

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

def plot_generic(Xs, Ys, x_labs, y_labs, legend = None, fname = None):
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

    if len(x_labs) <= len(Xs):
        x_labs = x_labs + [x_labs[-1]] * (len(Xs) - len(x_labs))
    if len(y_labs) <= len(Ys):
        y_labs = y_labs + [y_labs[-1]] * (len(Ys) - len(y_labs))

    fig = plt.figure(figsize = (4.5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    fmts = fmts[0:len(Xs[0])]
    for x, y, x_lab, y_lab, fmt in zip(Xs, Ys, x_labs, y_labs, fmts):
        line = ax.plot(x, y, fmt)
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        if re.match("log( ?|\()", x_lab, re.IGNORECASE): ax.set_xscale("log")
        if re.match("log( ?|\()", y_lab, re.IGNORECASE): ax.set_yscale("log")

    if legend is not None:
        if not isinstance(legend, (list, tuple)): legend = [legend]
        legend = ['\n'.join(wrap(l, 20)) if not l.startswith("$") else l for l in legend ]
        ax.legend(legend, fontsize = ax.xaxis.label.get_size()-1,
                    borderaxespad = 1.1)

    if fname is not None:
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
