import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.ticker import FormatStrFormatter

from sicm import analysis
from sicm.utils import utils

def make_colorbar(fig, cmap, fmt, levels, cmin, cmax):
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
    norm = mpl.colors.Normalize(vmin = cmin, vmax = cmax, clip = False)
    sm = plt.cm.ScalarMappable(norm = norm, cmap = cmap)
    sm.set_array([])
    # setting boundaries clips off colorbar extends that are not in current lelves
    cbar = fig.colorbar(sm, ticks = levels,
                        format = fmt, drawedges = False)
    return cbar

def _make_mask(x, y, x_range, y_range):
    """Make mask based on X,Y coordinates"""
    if x is not None:
        cond1 = np.logical_and(x > x_range[0], x < x_range[1])
    if y is not None:
        cond2 = np.logical_and(y > y_range[0], y < y_range[1])
    if x is not None and y is not None:
        mask = np.logical_and(cond1, cond2)
    else:
         mask = None
    return mask

def adjust_saliency(z, clip = None, x = None, y = None):
    """Adjust saliency

    Parameters
    ---------
    z: array-like
    clip: tuple
    x: array-like
    y: array-like

    Returns
    -------------
    z: array-like
        z, with saliency scaled in region selected by mask.
    """
    params = {  "x_range": [22.5, 32.5],
                "y_range": [0.0, 62.5],
                "thresh_quantile": 0.25}
    # Select region of array to look at
    mask = _make_mask(x, y, params["x_range"], params["y_range"])
    if mask is None:
        mask = [True] * len(z)

    ## compute scaling weights
    weight = 1 / z[mask]
    # thresh_quantile = np.quantile(weight[weight > 1.0], 0.35)
    # weight[weight > thresh_quantile] = thresh_quantile
    ## Don't scale anythign up
    weight[weight > 1.0] = 1.0
    # scale down, but avoid scaling too much
    thresh_quantile = np.quantile(weight[weight < 1.0], params["thresh_quantile"])
    weight[weight < thresh_quantile] = thresh_quantile
    # Apply clip to the data
    ## This may not be necessary, as data is clipped fro plotting anyway.
    if clip is not None:
        z[z > clip[1]] = clip[1]
        z[z < clip[0]] = clip[0]
    ## aply weights multiplier in selected region
    z[mask] *= weight
    return z, params

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
    """Plot 3d surface data with triangulation

    Parameters
    ----------------
    ax: matplotlib.pyplot.axis
    x: array-like
    y: array-like
    z: array-like
    cmap: str
        'afmhot' produces AFM colormap, otherwise go for e.g. 'binary'

    Returns
    ---------------
    surf: Axes3D.plot_trisurf
    """
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

def _plot_contour_3(ax, x, y, z, cmap = "afmhot", norm = None, levels = 10, z_aux = None):
    """Render 2D filled contour plot with triangulation

    Parameters
    --------
    ax: matplolib.pyplot.axis
    x: array-like
    y: array-like
    z: array-like
    cmap: str
        'afmhot' produces AFM colormap, otherwise go for e.g. 'binary'
    levels: int or array-like
        Levels of contour plot. If int, levels are determined by pyplot.
        (It should give the same result though)
    """
    if norm is None:
        conts = ax.tricontourf(x, y, z, cmap = cmap, levels = levels) # or greys
    else:
        # Extend 'both' replaces colors out of range by low-high range limits
        conts = ax.tricontourf( x, y, z, cmap = cmap, levels = levels,
                                norm = norm, extend = "both", alpha = 0.75)
    if z_aux is not None:
        # scale and convert to nm
        z_aux = (z_aux - np.nanmin(z_aux)) * 1e3
        conts_aux = ax.tricontour(x, y, z_aux, colors = "black", alpha = 0.1,
                                    linewidths = mpl.rcParams["lines.linewidth"]*0.2)
        plt.clabel( conts_aux, conts_aux.levels[::2], fmt = "%1.1f",
                    inline = True, fontsize = 4)

    ## Plotting data as an image. Not very nice.
    # z_ = np.reshape(z[:np.int(np.sqrt(z.shape[0]))**2], (np.int(np.sqrt(z.shape[0])), -1))
    # z_[1::2, :] = z_[1::2, ::-1]
    # conts = ax.imshow(  z_, cmap = cmap, norm = norm,
    #                     extent = [x.min(), x.max(), y.min(), y.max()],
    #                     alpha = 0.5)

    return conts

def plot_contour(ax, x, y, z, cmap = "afmhot", norm = None, levels = 10, z_aux = None):
    """Render 2D filled contour plot with triangulation

    Parameters
    --------
    ax: matplolib.pyplot.axis
    x: array-like
    y: array-like
    z: array-like
    cmap: str
        'afmhot' produces AFM colormap, otherwise go for e.g. 'binary'
    levels: int or array-like
        Levels of contour plot. If int, levels are determined by pyplot.
        (It should give the same result though)
    """
    conts = _plot_contour_3(ax, x, y, z, cmap, norm, levels, z_aux)
    return conts

def _plot_surface_contours( x, y, z, z_lab = "Z", ax = None, title = None,
                            fname = None, colors = None, center = True):
    """Plot 3D surface contours
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection = "3d")
        fig.tight_layout()
    else:
        fig = ax.get_figure()

    if center:
        if z_lab == "Z(um)" or z_lab.lower().startswith("z"):  # avoid centering current!
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
                n_levels = 10, z_aux = None):
    """Render 2D filled contour plot with triangulation

    Parameters
    --------
    ax: matplolib.pyplot.axis
    x: array-like
    y: array-like
    z: array-like
    fname: str
        Full path to an image to be created.
    cbar_lims: tuple
        2 values, limits of colorbar. If None, limits taken from data.
    center: bool
        Center x,y values around 0 and shift z to 0-origin?
    n_levels:
        Number of levels for contour plot
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.tight_layout()
    else:
        fig = ax.get_figure()

    if center:
        if z_lab == "Z(um)" or z_lab.lower().startswith("z"):  # avoid centering current!
            z = z - np.nanmin(z)
        x = x - np.nanmean(x)
        y = y - np.nanmean(y)

    if np.any(~np.isfinite(z)): # handle (expected) nans in data gracefully
        raise NotImplementedError("Sparse measurements not implemented!")
    else:
        with plt.style.context("seaborn-ticks"):
            fmt = "%.2e" if (np.max(np.abs(z)) < 1e-3 or np.min(np.abs(z)) > 1e3) else "%.3f"
            if any(cl is None for cl in cbar_lims):
                # Option 2: Variable colorbar, but better contrast for each slice
                conts = plot_contour(ax, x, y, z)
                cbar = fig.colorbar(conts, ticks = conts.levels, format = fmt, drawedges = False)
                # cbar.set_clim(*cbar_lims) # this appears not helpful
            else:
                # Option 1: Same colorbar for all slices
                levels_conts = np.linspace(cbar_lims[0], cbar_lims[1], n_levels)
                if cbar_lims[0] > z.min() or cbar_lims[1] < z.max():
                    levels_cbar = levels_conts
                else:
                    levels_cbar = np.linspace(z.min(), z.max(), n_levels)
                norm = mpl.colors.Normalize(vmin = cbar_lims[0], vmax = cbar_lims[1], clip = True)
                conts = plot_contour(ax, x, y, z, norm = norm, levels = levels_conts,
                                    z_aux = z_aux)
                cbar = make_colorbar(fig, conts.cmap, fmt, levels_cbar, *cbar_lims)

    cbar.ax.set_ylabel(z_lab)

    ax.set_xlabel('X(um)')
    ax.set_ylabel('Y(um)')
    plt.axis("scaled") # use this if data not square

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
    """Plot 3D surface contours

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
    _plot_surface_contours(x, y, z, z_lab, ax, center = center)

    # Save figure
    if fpath is not None:
        appex = "_surface"
        if z_lab.lower().startswith("cu"):
            appex += "Current"
        fname = utils.make_fname(fpath, appex)
        utils.save_fig(fname, ext = ".png")
    # Show
    plt.show()
