from argparse import ArgumentParser
from os.path import abspath

def float_or_none(x):
    if x is not None:
        x = float(x)
    return x

class SubArgParser(ArgumentParser):
    """Subclass of ArgumentParser with default values"""
    def __init__(self, *args, **kwargs):
        super(SubArgParser, self).__init__(*args, **kwargs)
        self._add_args()

    def _add_args(self):
        """Populate ArgParser object with default values"""
        self.description = "Scanning Ion Conductance Microscopy (SICM) Toolkit"
        self.add_argument("--version", action = "version", version = "%(prog)s 0.1.0")
        self.add_argument("-w", "--what", dest = "what", action = "store",
                        help = "What to plot?", choices = ["lockin", "scan"],
                        default = "scan", type = str.lower)
        self.add_argument(  "--correct", dest = "do_correct", action="store_true",
                            help = "Apply correction for jump in current?")
        self.add_argument(  "--constant_distance", dest = "is_constant_distance",
                            action="store_true", help = "Is the scan obtained at constant distance?")
        self.add_argument(  "--current", dest = "plot_current",
                            action="store_true", help = "Plot current on Z axis?")
        self.add_argument("datadir", action = "store", default = "./",
                        help = "Path to data directory [default='./']",
                        type = abspath)
        self.add_argument("exp_name", action = "store", type = str,
                        help = "Name of the experiment, for 'what=lockin' must also include extension")
        self.add_argument("-x", "--xrange", dest = "xrange", action = "store",
                        help= "Two numbers for low and high X defining range to retain.",
                        nargs = 2, type = float_or_none)

        self.add_argument("-y", "--yrange", dest = "yrange", action = "store",
                        help= "Two numbers for low and high Y defining range to retain.",
                        nargs = 2, type = float_or_none)
