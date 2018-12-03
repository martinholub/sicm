from argparse import ArgumentParser

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
        # self.add_argument("-i", "--input", dest = "input", action = "store",
        #                 help = "Path to file where replacements will be made", required = True,
        #                 type = path_normalizer)
        # self.add_argument("-w", "--what", dest = "what", action = "store",
        #                 type = str, default = "sheets", choices = ["sheets", "book"],
        #                 help = "Output individual sheets as csv or book as ods?")
