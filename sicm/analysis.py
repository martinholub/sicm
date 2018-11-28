import numpy as np
import copy

import matplotlib.pyplot as plt
import matplotlib as mpl

class Picker(object):
    """Object to collect information on clicked points"""
    def __init__(self):
        self.picks = []

    def onpick(self, event):
        """Event invoked on clikcing a point"""
        ind = event.ind[0]
        x, y, z = event.artist._offsets3d
        print("x:{}, y:{}, z:{}".format(x[ind], y[ind], z[ind]))
        self.picks.append({ind: (x[ind], y[ind], z[ind])})

def level_plane(X, Y, Z, is_debug = False, interactive = True):
    """Level Tilted Plane

    Sleection of pints for plane fitting is done interactively to deal with
    less predictable surface topography. The functions must be called from console (not from ipynb) to work properly.

    Parameters
    --------
    X, Y, Z : array-like
        1D arrays of point cooridnates in 3D space

    Returns
    ---------
    X_sq, Y_sq: array-like
        X, Y coordinates of points convertd to 2D matrix
    Z_sq_corr: array-like
        Z coordinates corrected for tilt

    References:
      *http://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
      *https://matplotlib.org/gallery/event_handling/ginput_demo_sgskip.html#sphx-glr-gallery-event-handling-ginput-demo-sgskip-py
      *https://stackoverflow.com/questions/21851114/can-matplotlib-pick-event-return-array-indeces-rather-than-values-or-pixels
      *https://matplotlib.org/api/collections_api.html#matplotlib.collections.Collection

    """
    # Reshape to square matrix, flip every second column
    a = np.int(np.sqrt(len(Z)))
    X_sq = np.reshape(X[:a**2], [a]*2); X_sq[1::2, :] = X_sq[1::2, ::-1]
    Y_sq = np.reshape(Y[:a**2], [a]*2); Y_sq[1::2, :] = Y_sq[1::2, ::-1]
    Z_sq = np.reshape(Z[:a**2], [a]*2); Z_sq[1::2, :] = Z_sq[1::2, ::-1]

    if interactive:
        # Select points interactively
        with mpl.rc_context(rc={'interactive': True}):
            fig = plt.figure(figsize = (6, 4))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_title("Click on 3 Points To Select Them")
            ax.scatter(X_sq.flatten(), Y_sq.flatten(), Z_sq.flatten(),
                        c = Z_sq.flatten(), marker = "^", picker = 5,
                        alpha = 0.3, cmap = "viridis")
            picker = Picker()
            cid = fig.canvas.mpl_connect("pick_event", picker.onpick)
            plt.show()
            input("[Press Enter once you have selected 3 points] \n")
            fig.canvas.mpl_disconnect(cid)
            # vals = plt.ginput(3, show_clicks = True) # just for 2D

        vals = [list(x.values())[0] for x in picker.picks]
        p1 = np.asarray(vals[0])
        p2 = np.asarray(vals[1])
        p3 = np.asarray(vals[2])
    else:
        p1 = np.asarray(((X_sq[1, 1], Y_sq[1, 1], Z_sq[1, 1])))
        p2 = np.asarray((X_sq[1, a-1], Y_sq[1,a-1], Z_sq[1, a-1]))
        p3 = np.asarray((X_sq[a-1, 1], Y_sq[a-1, 1], Z_sq[a-1, 1]))

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    # Compute tilt and correct for it
    Z_tilt = (d - a * X_sq - b * Y_sq) / c
    Z_sq_corr = Z_sq - (Z_tilt - np.min(Z_tilt))

    if is_debug:
        # Visualize selected points, their plane and the correctio
        with mpl.rc_context(rc={'interactive': True}):
            plt.style.use("seaborn")
            fig = plt.figure(figsize = (6, 4))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_title("Points Selected for Tilt Correction")

            ax.scatter(*zip(p1, p2, p3), s = 80, c = "r", marker="^")
            ax.plot_trisurf(X_sq.flatten(), Y_sq.flatten(), Z_sq.flatten(),
                            color = "gray", alpha = 0.2)
            ax.plot_trisurf(X_sq.flatten(), Y_sq.flatten(), Z_tilt.flatten(),
                            color = "red", alpha = 0.2)
            ax.plot_trisurf(X_sq.flatten(), Y_sq.flatten(), Z_sq_corr.flatten(),
                            color = "green", alpha = 0.2)
            ax.set_xticks([], []); ax.set_yticks([], []), ax.set_zticks([], []);
            plt.show()

    return (X_sq, Y_sq , Z_sq_corr)
