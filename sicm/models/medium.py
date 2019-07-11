import numpy as np
from sicm.plots import plots


class Medium(object):
    """Medium in the pipette"""
    def __init__(self, theta = np.nan, gamma = np.nan, rho = np.nan, kappa = np.nan):
        self.theta = theta # young's angle; degree
        self.gamma = gamma # Gas-Liaquid surface tension; N/m
        self.rho = rho #  differential density (e.g rho_water - rho_air); kg/m3
        self.kappa = kappa # conductivuty; S/m


# class Conductivity(object):
#     """Conductivity of electrolyte solution"""
#     def __init__(self):
#         pass

class Conductivity(object):
    """Analytical Model of Ion Current Through Pipette at Elevvated Temperature

    Implementation is as in Dmitry's notebook.
    """
    def __init__(self, c0 = 0.25, T0 = 298.15):
        self.c0 = c0 # Molar
        self.T0 = T0 # K

    @property
    def m_values(self):
        x = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        return np.asarray(x)

    @property
    def y_KCl_values(self):
        x = [   0.965,0.951,0.927,0.901,0.869,0.816,0.768,0.717,0.687,\
                0.665,0.649,0.636,0.626,0.617]
        return np.asarray(x)

    @property
    def M_salt(self):
        return 74.55e-3 # kg/mol

    @property
    def DCl_inf(self):
        return 2.032e-9

    @property
    def DK_inf(self):
        return 1.957e-9

    @property
    def F(self):
        return 96485.33212 # C/mol

    @property
    def R(self):
        return 8.31446261815324 # J / (K * mol)

    @property
    def d(self):
        """What is this parameter???"""
        return 1 # kg/L

    @property
    def c0_values(self):
        x = (self.d * self.m_values) / (1 + self.m_values * self.M_salt)
        return x

    def viscosity(self, value):
        """Viscosity at tempertature `value` """
        nu = 2.414e-5 * (10 ** (247.8 / (value - 140.)))
        return nu

    def intfun_KCl(self, values, is_deriv = False):
        """
        From Hammer & Wu
        TODO: Which paper is that???
        f(c0)
        """
        x = self.c0_values
        y = np.log10(self.y_KCl_values)
        if is_deriv:
            return np.interp(values, x[1:], np.diff(y))
        else:
            return np.interp(values, x, y)
        return

    def DK_c(self, value):
        # value = self.c0
        grad = self.intfun_KCl(value, is_deriv = True)
        return self.DK_inf * (1 + value * grad)

    def DCl_c(self, value):
        # value = self.c0
        grad = self.intfun_KCl(value, is_deriv = True)
        return self.DCl_inf * (1 + value * grad)

    def DK(self, T):
        """Diffusivity of K at c0 as function of temperature"""
        val = self.DK_c(self.c0)
        val *= (T * self.viscosity(self.T0)) /(self.T0 * self.viscosity(T))
        return val

    def DCl(self, T):
        """Diffusivity of Cl at c0 as function of temperature"""
        val = self.DCl_c(self.c0)
        val *= (T * self.viscosity(self.T0)) /(self.T0 * self.viscosity(T))
        return val


    def uK(self, T):
        """Mobility of K at c0 as function of temperature"""
        val = self.F / (self.R * T) * self.DK(T)
        return val

    def uCl(self, T):
        """Mobility of Cl at c0 as function of temperature"""
        val = self.F / (self.R * T) * self.DCl(T)
        return val

    def conductivity(self, T = None):
        """Solution conductivity at given temperature in S/m"""
        try:
            if not T: T = self.T0
        except ValueError as e:
            if not T.size: T = self.T0
        kappa = self.F * self.c0 * 1e3
        kappa *= (self.uK(T) + self.uCl(T))
        return kappa

    def kappa(self, T = None):
        """Alias for conductivity"""
        return self.conductivity(T)
