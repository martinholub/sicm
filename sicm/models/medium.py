import numpy as np
from sicm.plots import plots
from scipy import interpolate as interp


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

    References
    -----------
    [1]  https://doi.org/10.1063/1.3253108
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
        """ Increment function
        """
        x = self.c0_values
        y = np.log10(self.y_KCl_values)

        # if is_deriv:
        #     return np.interp(values, x[1:], np.diff(y))
        # else:
        #     return np.interp(values, x, y)

        ## Option # 1
        # unispl = interp.UnivariateSpline(x, y)
        # val = unispl(values)
        # if is_deriv:
        #     unispl_der = unispl.derivative()
        #     val = unispl_der(values)

        ## Option # 2
        coeffs = np.polynomial.hermite.hermfit(x, y, deg = 3)
        val = np.polynomial.hermite.hermval(values, coeffs)
        if is_deriv:
            coeffs_der = np.polynomial.hermite.hermder(coeffs)
            val = np.polynomial.hermite.hermval(values, coeffs_der)

        # ## Option # 3
        # interpfun = interp.interp1d(x, y, kind = "linear")
        # val = interpfun(values)
        # if is deriv:
        #     raise NotImplementedError()
        #
        # ## Option 4
        # coeffs = np.polyfit(x, y, 3)
        # val = np.polyval(coeffs, values)
        # if is_deriv:
        #     der = np.polyder(np.poly1d(coeffs), m = 1)
        #     val = der(values)
            # coeffs = der.coeffs[::-1]
            # val = np.sum((p*values**i for i, p in enumerate(coeffs)))



        return val

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

    def density(self, T):
        """Density as function of temperature

        Using equation (6) or (1) from [1].

        References
        --------------
        [1]  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4909168/
        """
        ## Eq. (6) from [1]
        rho = 999.848847
        rho += 6.337563e-2 * T
        rho -= 8.523829e-3 * T**2
        rho += 6.943248e-5 * T**3
        rho -= 3.821216e-7 * T**4

        ## Eq (1) from [1]:
        rho = 999.83952
        rho += 16.945176 * T
        rho -= 7.9870401e-3 * T**2
        rho -= 46.170461e-6 * T**3
        rho += 105.56302e-9 * T**4
        rho -= 280.54253e-12 * T**5
        rho /= (1 + 16.897850e-3 * T)

        ## Eq from COMSOL
        rho = 838.466135
        rho += 1.40050603 * T**1
        rho -= 0.0030112376 * T**2
        rho += 3.71822313e-7 * T**3
        return rho

    def specific_heat(self, T):
        """Specific heatof water as function of temperature

        Function is from COMSOL
        """
        cp = 12010.1471
        cp -= 80.4072879 * T**1
        cp += 0.309866854 * T**2
        cp -= 5.38186884e-4 * T**3
        cp += 3.62536437e-7 * T**4
        return cp

    def thermal_conductivity(self, T):
        """Thermal conductivity of water as function fo T

        Function is from Comsol
        """
        k = -0.869083936
        k += 0.00894880345 * T**1
        k -= 1.58366345e-5 * T**2
        k += 7.97543259e-9 * T**3
        return k

    def thermal_diffusivity(self, T):
        """Thermal diffusivity of medium given temperature T """
        
        alpha = self.thermal_conductivity(T)
        alpha /= (self.specific_heat(T) * self.density(T))
        return alpha


    def relative_density(self, T):
        rhoT = self.density(T)
        rhoT0 = self.density(self.T0)
        return rhoT / rhoT0

    def conductivity(self, T = None):
        """Solution conductivity at given temperature in S/m"""
        try:
            if not T: T = self.T0
        except ValueError as e:
            if not T.size: T = self.T0
        kappa = self.F * self.c0 * 1e3
        kappa *= (self.uK(T) + self.uCl(T))
        kappa *= (self.density(T)) / (self.density(self.T0))
        return kappa

    def kappa(self, T = None):
        """Alias for conductivity"""
        return self.conductivity(T)
