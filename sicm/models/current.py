import numpy as np
from sicm.plots import plots

from .medium import Conductivity

class TestCurrent(object):
    """Utility class to calculate current

    Using equations as in Dmitry's notebook.
    """
    def __init__(self, U, r_tip, alpha):
        self.U = U
        self.r_tip = r_tip
        self.alpha = alpha # intrnal semi-angle

    def calculate_current(self, T, T0, c0):
        kappa = Conductivity(c0, T0).kappa(T)
        cot_semia = 1 / np.tan(np.deg2rad(self.alpha/2))
        r_tot = (1 / kappa) * ((1 + cot_semia) / (2 * np.pi * self.r_tip))
        i = self.U / r_tot
        return i

    def plot_current(self, T = None, T0 = 298.15, c0 = 0.5, plot_relative = False,
                    add_data = {}, **kwargs):

        x_lab = ["T [K]"]
        y_lab = ["I [nA]"]

        if T is not None:
            i = self.calculate_current(T, T0, c0) * 1e9
            i_t0 = self.calculate_current(T0, T0, c0) * 1e9

            if plot_relative:
                i = i / i_t0
                y_lab = [r"$\rm{I_{rel}}}$ [nA]"]

            x = [T]; y = [i]; leg = ["model"]
        else:
            x = []
            y = []
            leg = []

        if add_data:
            x += add_data["x"]
            y += add_data["y"]
            leg += add_data["leg"]

        plots.plot_generic(x, y, x_lab, y_lab, leg, **kwargs)
