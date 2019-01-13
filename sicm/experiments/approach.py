
import numpy as np
import os
from matplotlib import pyplot as plt
from copy import deepcopy

from sicm import plots
from sicm.utils import utils
from .experiment import Experiment
from ..measurements.signal import Signal

class Approach(Experiment):
    def __init__(self, datadir, exp_name):
        super(Approach, self).__init__(datadir, exp_name)
        # TODO: handle data, guessid params

    def plot(self, sel = None):
        plots.plot_sicm(self.dsdata, sel, "Approach", self.name, self.date)
