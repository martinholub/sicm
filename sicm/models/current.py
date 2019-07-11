import numpy as np
from sicm.plots import plots

from .medium import Conductivity

class TestCurrent(object):
    """Temporary utility function to calculate current

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

    def plot_current(self, T, T0 = 298.15, c0 = 0.5, plot_relative = False):
        i = self.calculate_current(T, T0, c0)
        i_t0 = self.calculate_current(T0, T0, c0)

        if not plot_relative:
            plots.plot_generic([T], [i], ["T [K]"], ["I [nA]"])

        else:
            i = i / i_t0
            plots.plot_generic([T], [i], ["T [K]"], [r"$\rm{I_{rel}}}$ [nA]"])
