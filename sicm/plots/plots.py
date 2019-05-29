
import numpy as np
import os
from copy import deepcopy
from textwrap import wrap
import itertools
import re

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.ticker import FormatStrFormatter

from sicm import analysis
from sicm.utils import utils

def _set_rcparams():
    plt.style.use("seaborn-white")
    params = {  "font.family": "serif",
                "font.weight": "normal",
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "xtick.bottom": True,
                "xtick.direction": "in",
                "ytick.left": True,
                "ytick.direction": "in"}
    mpl.rcParams.update(params)

def plot_mock(ax):
    """Render a mock plot"""
    ax.text(x=0.5, y=0.5, s = "No Data Available",
            horizontalalignment = "center", wrap = "True",
            fontsize = "xx-large",
            verticalalignment="center", transform=ax.transAxes)

def _plot_mark_idxs(ax, x, y, idxs = None):
    """Mark index in data on plot
    """
    if idxs is not None:
        try:
            ax.plot(x[idxs], y[idxs],
                        color = "red", marker = ".", linestyle = "None",
                        alpha = 0.25)
        except Exception as e:
            pass
    else:
        try:
            ax.plot(x, y,
                        color = "red", marker = ".", linestyle = "None",
                        alpha = 0.25)
        except Exception as e:
            pass

def _xylabel_getter(x_key, y_key):
    """Repurpose X,Y variable names for axes labels"""
    x_lab = x_key.replace("(", " [").replace(")", "]")
    x_lab = re.sub(r'[0-9]+', '', x_lab)
    y_lab = y_key.replace("(", " [").replace(")", "]")
    y_lab = re.sub(r'[0-9]+', '', y_lab)

    return x_lab, y_lab

def _sicm_subplot(ax, data, sel, x_key, y_key, idxs = None):
    """Render x,y selection from data"""
    # Obtain label name from variable name
    x_lab, y_lab = _xylabel_getter(x_key, y_key)
    try:
        # Mark idxs points
        _plot_mark_idxs(ax, data[x_key], data[y_key], idxs)
        ax.plot(data[x_key][sel], data[y_key][sel])
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
    except KeyError as e:
        plot_mock(ax)

def plot_sicm(  result, sel, title = "SICM Plot", exp_name = None,
                date = "00/00/0000", fname = None, idxs = None, dsdata = None):
    """Generic SICM plot"""
    # Check that data for plotting is ready
    if sel is None:
        try:
            sel = np.arange(0, len(result["LineNumber"]))
        except KeyError as e:
            sel = np.arange(0, len(next(iter(data.values()))))
    elif idxs is not None:
        raise NotImplementedError("plot_sicm: Cannot supply `idxs` and `sel` simultaneously.")

    if "time(s)" not in result.keys():
        result["time(s)"] = np.cumsum(result["dt(s)"])

    # Decide which keys to plot against which
    x_keys = ["time(s)"]*2  + ["LineNumber", "time(s)"]*3 + ["X(um)", "time(s)"]
    y_keys = ["V1(V)", "Current1(A)"] + ["Z(um)"]*2 + ["X(um)"]*2 + ["Y(um)"]*3 + ["LineNumber"]
    assert len(x_keys) == len(y_keys)

    # Set-up Plotting
    plt.style.use("seaborn-white")
    nrows = int(np.ceil(len(x_keys)/2))
    fig, axs = plt.subplots(nrows = nrows, ncols = 2, figsize = (10, 5 * nrows))
    axs = axs.flatten()
    title = title + "\n" + "{} @ {}".format(exp_name, date)
    fig.suptitle(title, size = 18, y = 0.94)

    # Render all plots
    for i, ax in enumerate(axs):
        _sicm_subplot(ax, result, sel, x_keys[i], y_keys[i], idxs)

    fname_lockin = None
    if fname is not None:
        utils.save_fig(fname)
        fname_lockin = utils.make_fname(fname, "Lockin", ext = ".png")

    # Plot also results from lockin
    subkeys =  [("time(s)", "LockinPhase"), ("time(s)", "LockinAmplitude")]
    sel_keys = set([y for x in subkeys for y in x])
    subresult = {k:v[sel] for k,v in result.items() if k in sel_keys}
    if dsdata is not None:
        dsdata = {k:v for k,v in dsdata.items() if k in sel_keys}
    plot_lockin(subresult, subkeys, name = exp_name, date = date, fname = fname_lockin,
                idxs = idxs, dsdata = dsdata)

def plot_lockin(data = {}, keys = [("frequency", "r")], date = None, name = None,
                fname = None, xlog = False, idxs = None, dsdata = None):
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

    plt.style.use("seaborn-white")
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

            if dsdata is not None:
                _plot_mark_idxs(axs[i], dsdata[k[0]], dsdata[k[1]])
            else:
                _plot_mark_idxs(axs[i], data[k[0]], data[k[1]], idxs)

            if xlog:
                axs[i].set_xscale("log")
                axs[i].set_xlabel("log " + labels[k[0]])
            # axs[i].set_title(" ".join(labels[k[1]].split(" ")[:-1]))
        except KeyError as e:
            plot_mock(axs[i])

    if fname is not None:
        utils.save_fig(fname)

    plt.show()

def plot_generic(Xs, Ys, x_labs, y_labs, legend = None, fname = None, fmts = None,
                ax = None, text = None, text_loc = (0.1, 0.1)):
    """Generic ploting function

    This is an attempt of generic function that produces publication quality
    plots. Parameters have their usual meanings.
    """
    # set plotting style
    _set_rcparams()

    if fmts is None:
        fmts_prod= itertools.product(["k"], ["-", "--", ":", "-."])
        fmts = ["".join(x) for x in fmts_prod]

    if len(x_labs) <= len(Xs):
        x_labs = x_labs + [x_labs[-1]] * (len(Xs) - len(x_labs))
    if len(y_labs) <= len(Ys):
        y_labs = y_labs + [y_labs[-1]] * (len(Ys) - len(y_labs))

    if ax is None:
        fig = plt.figure(figsize = (4.5, 4.5))
        ax = fig.add_subplot(1, 1, 1)
    fmts = fmts[0:len(Xs[0])]
    for x, y, x_lab, y_lab, fmt in zip(Xs, Ys, x_labs, y_labs, fmts):
        try:
            line = ax.plot(x, y, fmt)
        except ValueError as e:
            try: # allow grey colors in dirty way
                line = ax.plot(x, y, linestyle = fmt[0], color = fmt[1:])
            except ValueError as e:
                try:
                    line = ax.plot(x, y, marker = fmt[0], color = fmt[1:], linestyle = "None")
                except Exception as e:
                    raise e
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        if re.match("log( ?|\()", x_lab, re.IGNORECASE): ax.set_xscale("log")
        if re.match("log( ?|\()", y_lab, re.IGNORECASE): ax.set_yscale("log")

    if legend is not None and legend != "":
        if not isinstance(legend, (list, tuple)): legend = [legend]
        legend = ['\n'.join(wrap(l, 21)) if not l.startswith("$") else l for l in legend]
        ax.legend(legend, fontsize = ax.xaxis.label.get_size()-1,
                    borderaxespad = 1.1)
        # bbox_to_anchor=(1.01,1), loc="upper left" # if you need the legend outside
    elif len(Xs) > 1:
        plt.legend(range(len(Xs)))

    if text is not None:
        ax.text(text_loc[0], text_loc[1], text, transform = ax.transAxes,
                color = "black")

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
    _set_rcparams()

    if x_labs is None:
        try:
            x_labs = np.arange(1, len(x[0])+1).tolist()
        except ValueError as e:
            x_labs = np.arange(1, len(x)+1).tolist()
    if not isinstance(x_labs, (list, tuple, np.ndarray)): x_labs = [x_labs]

    fig = plt.figure(figsize = (4.5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    bxplt = ax.boxplot( x, labels = x_labs, showmeans = True, meanline = True,
                        medianprops = {"color": "black", "linestyle": '-.'},
                        meanprops = {"color": "gray", "linestyle": ':'})
    ax.set_ylabel(y_lab)
    if np.mean(x) < 0.05:
        ax.set_yscale("log")

    # legend
    if legend is not None:
        if not isinstance(legend, (list, tuple)): legend = [legend]
        legend = ['\n'.join(wrap(l, 20)) for l in legend]
        # `handletextpad=-2.0, handlelength=0` hides the marker in legend [1]
        ax.legend(legend, fontsize = ax.xaxis.label.get_size()-2)

    if fname is not None:
        utils.save_fig(fname)
    # recover plotting style
    mpl.rcParams.update(mpl.rcParamsDefault)


def errorplot_generic(  Xs, Ys, Y_errs, x_lab = None, y_lab = None, legend = None,
                            fname = None):
    """Generic Errorbar plot function
    """

    # set plotting style
    _set_rcparams()

    fmts_prod= itertools.product(["k"], ["-", "--", ":", "-."], ["^", "s", "o"])
    fmts = ["".join(x) for x in fmts_prod]

    fig = plt.figure(figsize = (4.5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    handles = []
    for i, (x, y, yerr) in enumerate(zip(Xs, Ys, Y_errs)):
        ebars = ax.errorbar(x, y, yerr, fmt = fmts[i],
                            capsize = 2, elinewidth = 1, ecolor = "gray", capthick = 1)
        handles.append(ebars[0])

    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    ax.set_xscale("log")

    # legend
    if legend is not None:
        if not isinstance(legend, (list, tuple)): legend = [legend]
        legend = ['\n'.join(wrap(l, 20)) for l in legend]
        # `handletextpad=-2.0, handlelength=0` hides the marker in legend [1]
        ax.legend(handles, legend, fontsize = ax.xaxis.label.get_size()-2)

    if fname is not None:
        utils.save_fig(fname)
    # recover plotting style
    mpl.rcParams.update(mpl.rcParamsDefault)
