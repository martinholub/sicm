from sicm import io, analysis, filters
from sicm.plots import plots
from sicm.utils import utils
from sicm.models.model import Model
from sicm.models.medium import Medium
from sicm.mathops.mathops import is_null

import numpy as np
import timeit
from scipy import integrate
from copy import deepcopy

import matplotlib.pyplot as plt

class Pipette(object):
    """"Pipette Geometry Model"""
    def __init__(self, d_tip, alpha = None, length = None, d_body = None):
        self.d_tip  = d_tip # inner diameter of the tip; m
        self.alpha = alpha # tip opening angle; degree
        self.length = length # length of pipette; m
        self.d_body = d_body # inener diameter of the capillary; m

    @property
    def d_body(self):
        return self._d_body
    @d_body.setter
    def d_body(self, value):
        if is_null(value):
            try:
                value = self._calculate_upper_diameter()
            except Exception as e:
                import pdb; pdb.set_trace()
                value = np.nan
        else:
            pass
        self._d_body = value

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        if is_null(value):
            try:
                value = self._calculate_pipette_length()
            except Exception as e:
                value = np.nan
        else:
            pass
        self._length = value


    def _calculate_upper_diameter(self):
        """Calculate inner diameter at the top of the pipette"""
        value = self.length * np.tan(np.deg2rad(self.alpha))
        value += np.abs(self.d_tip/2)
        value *= 2
        return value

    def _calculate_pipette_length(self):
        """Calculate pipette length"""
        value = np.tan(np.deg2rad(self. alpha))
        value /= np.abs(self.d_body - self.d_tip)
        return value


    def _calculate_resistance_inside(self, kappa, length):
        """Calculate resistance inside a pore by equation (4)"""
        # solid angle; eq. 5
        omega = 2 * np.pi * (1 - np.cos(np.deg2rad(self.alpha)))
        prefactor = (1 / kappa) * (1 / omega)

        def integrand(x):
            val = 1 / (x ** 2)
            return val

        lim1 = (self.d_tip / 2) / np.sin(np.deg2rad(self.alpha))
        lim2 = length + lim1
        if not isinstance(lim2, (list, tuple, np.ndarray)): lim2 = np.asarray([lim2])
        r_i = []

        try:
            for l2 in lim2:
                ret = integrate.quad(integrand, lim1, l2)[0]
                ret *= prefactor
                r_i.append(ret)
            r_i = np.asarray(r_i)
        except Exception as e:
            raise e

        return r_i

    def _calculate_resistance_outside(self, kappa, length):
        """Calculate resistance outside a pore by equation (3)"""
        prefactor  = 1 / (2 * np.pi * kappa)

        def integrand(x):
            val = 1 / (x ** 2)
            return val

        ## This is not working, donkt know why
        # try:
        #     r_o = integrate.quad(integrand, self.d_tip / 2, 1)[0]
        #     r_o *= prefactor
        # except Exception as e:
        #     raise e

        r_o = prefactor * (1 / (self.d_tip / 2))

        try:
            r_o = np.asarray([r_o] * len(length))
        except Exception as e:
            r_o = np.asarray([r_o])

        return r_o


    def calculate_resistance(self, medium, length = None):
        """Calculate total resistance of the pore

        Parameters
        --------------
        medium: Medium class
            Class implementing property kappa (conductivity)
        length: float or array-like
            The length of the pipette to consider in the integration

        Returns
        -----------
        R: float or array-like
            The total resistance of the pipette
        """
        kappa = medium.kappa
        try:
            length_ = self.length if not length else length
        except ValueError as e:
            length_ = self.length if not length.size else length


        R_i = self._calculate_resistance_inside(kappa, length_)
        R_o = self._calculate_resistance_outside(kappa, length_)

        R = R_i + R_o
        return R

    def plot_resistance(self, medium, length = None, **kwargs):
        """Plot resistance of the pipette

        Parameters
        -------
        (see `calculate_resistance`)
        **kwargs: dict
            Optional K:V pairs of args passed to plotting function
        """

        R = self.calculate_resistance(medium, length)
        # using higher length than 1e-1 usually fails with complaints
        R_inf = self.calculate_resistance(medium, 1e-1)

        x = [length / self.d_tip]
        y = [R / R_inf]

        x_lab = [r"L / $\rm{d_{tip}}$"]
        # y_lab = [r"$\rm{R_{tot}}$ [$\rm{\Omega}$]"]
        y_lab = [r"$\rm{R_{tot}^{rel}}$"]

        plots.plot_generic(x, y, x_lab, y_lab, **kwargs)
