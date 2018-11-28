
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits import mplot3d
import matplotlib.tri as tri

from sicm import analysis

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

def plot_surface(result):
    """Plot surface as contours and 3D"""

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
