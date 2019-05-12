import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.ticker import FormatStrFormatter

from sicm import analysis
from sicm.utils import utils

def make_colorbar(fig, cmap, levels, cmin, cmax):
    """Make nice colorbar

    Parameters
    ------------
    fig: matplotlib.figure.Figure
    cmap: matplotlib.colors.Colormap
    levels: array-like
        Levels of countour plot
    cmin: float
        Minimum relative value for colorbar color
    cmax: float
        Maximum relative value for colorbar color

    Returns
    ------------
    cbar: matplotlib.pyplot.Colorbar

    References
    ---------
    [1] https://stackoverflow.com/questions/44498631/continuous-colorbar-with-contour-levels
    """
    norm = mpl.colors.Normalize(vmin = cmin, vmax = cmax)
    sm = plt.cm.ScalarMappable(norm = norm, cmap = cmap)
    sm.set_array([])
    # setting boundaries clips off colorbar extends that are not in current lelves
    cbar = fig.colorbar(sm, ticks = levels,
                        format = "%.3f", drawedges = False)
    return cbar

def _plot_surface_1(ax, x, y, z, cmap):
    """Plot 3d surface data without interpolation

    This could work if data was always square. This is not the case though.

    """
    # Reshape to square matrix
    a = np.int(np.sqrt(len(z)))
    x_sq = np.reshape(x[:a**2], [a]*2)
    y_sq = np.reshape(y[:a**2], [a]*2)
    z_sq = np.reshape(z[:a**2], [a]*2)
    # Flip every second column ? Is this needed???
    x_sq[1::2, :] = x_sq[1::2, ::-1]
    y_sq[1::2, :] = y_sq[1::2, ::-1]
    z_sq[1::2, :] = z_sq[1::2, ::-1]
    surf = ax.plot_surface(x_sq, y_sq, z_sq, cmap = cmap)

    return surf

def _plot_surface_2(ax, x, y, z, cmap):
    """Plot 3d surface data without interpolation

    This will not work because z is not f(x,y)
    """
    xx, yy, zz = np.meshgrid(x, y, z)
    surf = ax.plot_surface(xx, yy, zz, cmap = cmap)
    return surf

def _plot_surface_3(ax, x, y, z, cmap):
    """ Plot 3d surface data with triangulation
    """
    trsf = ax.plot_trisurf(x, y, z, cmap = cmap)
    return trsf

def _plot_surface(ax, x, y, z, cmap):
    surf = _plot_surface_3(ax, x, y, z, cmap)
    return surf


def _plot_contour_1(ax, x, y, z, cmap, norm):
    """ Plot data as a 2D contour plot.

    This may work if there is only few nans. This is often not the case.
    """
    # A) would work if just few nans
    # https://github.com/matplotlib/matplotlib/issues/10167
    triang = mpl.tri.Triangulation(x, y)  # Delaunay triangulation of all points
    point_mask = ~np.isfinite(z)   # Points to mask out.
    tri_mask = np.any(point_mask[triang.triangles], axis = 1)  # Triangles to mask out.
    triang.set_mask(tri_mask)
    levels = np.linspace(np.nanmin(z), np.nanmax(z), 10)
    conts = ax.tricontourf(triang, z, cmap = cmap, levels = levels, norm = norm)
    return conts

def _plot_contour_2(ax, x, y, z, cmap, norm):
    """ Plot data as a 2D contour plot

    This would work for square data only
    """
    # Reshape to square matrix
    a = np.int(np.sqrt(len(z)))
    x_sq = np.reshape(x[:a**2], [a]*2)
    y_sq = np.reshape(y[:a**2], [a]*2)
    z_sq = np.reshape(z[:a**2], [a]*2)
    # Flip every second column ? Is this needed???
    x_sq[1::2, :] = x_sq[1::2, ::-1]
    y_sq[1::2, :] = y_sq[1::2, ::-1]
    z_sq[1::2, :] = z_sq[1::2, ::-1]
    conts = ax.contourf(x_sq, y_sq, z_sq, cmap =cmap, levels = 10, norm = norm)
    return conts

def _plot_contour_3(ax, x, y, z, cmap = "gray", norm = None, levels = 10):
    """ Plot surface plot with triangulation
    """
    if norm is None:
        conts = ax.tricontourf(x, y, z, cmap = cmap, levels = levels) # or greys
    else:
        conts = ax.tricontourf(x, y, z, cmap = cmap, levels = levels, norm = norm)
    return conts

def plot_contour(ax, x, y, z, cmap = "gray", norm = None, levels = 10):
    conts = _plot_contour_3(ax, x, y, z, cmap, norm, levels)
    return conts


def plot_projection(x, y, z, z_lab = "Z", ax = None, title = None,
                    fname = None, colors = None, center = True):
    """Plot projection of 3D data on a surface
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection = "3d")
        fig.tight_layout()
    else:
        fig = ax.get_figure()

    if center:
        if z_lab == "Z(um)":  # avoid centering current!
            z = z - np.nanmin(z)
        x = x - np.nanmean(x)
        y = y - np.nanmean(y)

    trsf = _plot_surface(ax, x, y, z, cmap = "binary")
    ax.set_xlabel('X(um)')
    ax.set_ylabel('Y(um)')
    ax.set_zlabel(z_lab)

    if colors is not None:
        trsf.set_array(colors)
        trsf.autoscale()

    # Set descriptive title
    if title is not None:
        ax.set_title(title)
    # Save figure
    if fname is not None:
        utils.save_fig(fname, ext = ".png")

    # Explicitly close all figures if already too many;
    if len(plt.get_fignums()) > 3:
        plt.close('all')

def plot_slice( x, y, z, z_lab = "Z", ax = None, title = None,
                fname = None, cbar_lims = (None, None), center = True,
                n_levels = 10):
    """Plot a single slice"""


    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.tight_layout()
    else:
        fig = ax.get_figure()

    if center:
        if z_lab == "Z(um)":  # avoid centering current!
            z = z - np.nanmin(z)
        x = x - np.nanmean(x)
        y = y - np.nanmean(y)

    if np.any(~np.isfinite(z)): # handle (expected) nans in data gracefully
        raise NotImplementedError("Sparse measurements not implemented!")
    else:
        with plt.style.context("seaborn-ticks"):
            if any(cl is None for cl in cbar_lims):
                # Option 2: Variable colorbar, but better contrast for each slice
                conts = plot_contour(ax, x, y, z)
                cbar = fig.colorbar(conts, ticks = conts.levels, format = "%.3f", drawedges = False)
                # cbar.set_clim(*cbar_lims) # this appears not helpful
            else:
                # Option 1: Same colorbar for all slices
                levels = np.linspace(z.min(), z.max(), n_levels)
                norm = mpl.colors.Normalize(vmin = cbar_lims[0], vmax = cbar_lims[1])
                conts = plot_contour(ax, x, y, z, norm = norm, levels = levels)
                cbar = make_colorbar(fig, conts.cmap, conts.levels, *cbar_lims)

    cbar.ax.set_ylabel(z_lab)

    ax.set_xlabel('X(um)')
    ax.set_ylabel('Y(um)')

    # Set descriptive title
    if title is not None:
        ax.set_title(title)
    # Save figure
    if fname is not None:
        utils.save_fig(fname, ext = ".png")

    # Explicitly close all figures if already too many;
    if len(plt.get_fignums()) > 3:
        plt.close('all')

def plot_surface_contours(x, y, z, z_lab = "z", fpath = None, center = False):
    """Plot view of surface

    Parameters
    -------------
    x,z,y: array-like
        x,y coordinates and corresponding z-values
    z_lab: str
        z-axis label
    fpath: str
        A full-path to an image that will be created.
    center: bool
        Centers x,y around 0 and shifts origin of z to 0.
    """
    plt.style.use("seaborn")
    fig = plt.figure(figsize = (12, 10))
    # fig.tight_layout()

    # Filled countour with triangulation
    ax = fig.add_subplot(2, 2, 1)
    plot_slice(x, y, z, z_lab, ax, center = center)

    # Surface in 3D projection
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    plot_projection(x, y, z, z_lab, ax, center = center)

    # Save figure
    if fpath is not None:
        fname = utils.make_fname(fpath, "_surface")
        utils.save_fig(fname, ext = ".png")
    # Show
    plt.show()
