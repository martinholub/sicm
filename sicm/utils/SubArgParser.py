from argparse import ArgumentParser
from os.path import abspath

def float_or_none(x):
    if x is not None:
        x = float(x)
    return x

def hop_or_bulk_str(x):
    if not isinstance(x, (str, )): return "hop"

    if x.lower().startswith("b"):
        return "bulk"
    elif x.lower().startswith("h"):
        return "hop"
    else:
        return "hop"

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
        self.add_argument(  "--slices", dest = "plot_slices",
                            action="store_true", help = "Plot slices at various Z?")
        self.add_argument(  "--n_slices", "-n", dest = "n_slices", default = 10,
                            type = int, action="store", help = "How many slices to plot?")
        self.add_argument(  "--thickness", "-t", dest = "thickness", default = 0.9,
                            type = float, action="store", help = "Relative thickness of each slice.")
        self.add_argument(  "--center", dest = "center",
                            action="store_true", help = "Center values around/at 0?")

        self.add_argument(  "--scale",  dest = "scale", type = hop_or_bulk_str,
                            action="store", default = "hop",
                            help = "Scale by 'bulk' or 'hop'?")

        self.add_argument(  "-c", "--clip", dest = "clip", action="store",
                            nargs = 2, type = float_or_none,
                            help="Two numbers for low/high relative current to clip to.")
        self.add_argument(  "-l", "--levels", dest = "n_levels", action="store",
                            type = int, default = 10,
                            help="How many levels to split relative current to?")

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

        self.add_argument("-z", "--zrange", dest = "zrange", action = "store",
                        help= "Two numbers for low and high Z defining range to retain.",
                        nargs = 2, type = float_or_none)
