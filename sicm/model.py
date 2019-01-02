from sicm import io, analysis, plots, filters
from sicm.utils import utils

import numpy as np

class SICMModel(object):
    """SICM Model

    Equations (1)-(3) from Chen2012.
    """
    def __init__(self, z, U, h, kappa, r_p, r_i, r_o):
        self.R_p = self._calculate_pipette_resistance(h, kappa, r_p, r_i)
        self.R_ac = self._calculate_access_resistance(z, kappa, r_o, r_i)
        self.I = self._calculate_current(U)
        self.z = z
        self.r_i = r_i
        # Assing also other parameters
        self.r_p = r_p
        self.r_o = r_o
        self.h = h
        self.U = U
        self.kappa = kappa


    def _calculate_pipette_resistance(self, h, kappa, r_p, r_i):
        """tba"""
        R_p = h / (kappa*r_p*r_i*np.pi)
        return R_p

    def _calculate_access_resistance(self, z, kappa, r_o, r_i):
        """tba"""
        R_ac = 1.5*np.log(r_o/r_i) / (z*kappa*np.pi)
        return R_ac

    def _calculate_current(self, U):
        """tba"""
        I = U / (self.R_p + self.R_ac)
        return I

    def plot(self):
        x = self.z/(2*self.r_i) # scale by diameter
        y = np.abs(self.I)*1e9 # convert to nanoAmps
        leg = "Current-distance relation @ T=298.15K"
        plots.plot_generic(x, y, r"$z/d$", r"$I\ [nA]$", leg, "I vs. z curve")

    def fit_wrapper(self):
        """A wraper to define variables in the fitting function scope"""
        # dont treat these as free variables
        r_i = self.r_i
        r_o = self.r_o
        U = self.U
        # Assume you know what a good guess is
        guess = (self.r_p, self.h, self.kappa)

        def _inverse_fit(i, *params):
            """Fit function that yields distance for value of current"""
            r_p, h, kappa = params
            R_p = h / (kappa*r_p*r_i*np.pi)
            d = (i * 1.5 * np.log(r_i/r_o)*r_p*r_i) / (h*(i-U/R_p))
            return d

        return _inverse_fit, guess


    def fit_approach(self):
        """TODO: Continue here by writing the fitting function

        Goals is to have fun that gives d for value of I. The I will be obtained
        from experiments and will always have corresponding phase change attached to it.
        You will porobably need to think a bit what is the quantity you are relating to.
        Else, it should be straightforward."""
