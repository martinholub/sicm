from sicm import io, analysis, plots, filters
from sicm.utils import utils

class Scan(object):
    """"""
    def __init__(self):
        self.raw_result = result
        self.exp_name = exp_name
        self.date = date
        self.result, self.idxs = self.downsample_to_linenumber()
        self.sel = sel

    def downsample_to_linenumber(self):
        uniqs, cnts = np.unique(self.raw_result["LineNumber"], return_counts=True)
        linenos = np.arange(5, max(uniqs), 3)
        result, idxs = io.downsample_to_linenumber(self.raw_result, linenos)
        return(result, idxs)

    def analyze_hopping_scan(self):
        """"""
        sel = self.sel
        if sel is None:
            sel = np.arange(0, len(result["LineNumber"])//1)
        plots.plot_hopping_scan(self.result, sel, self.exp_name, self.date)

    def analyze_approach(self):
        """"""
        #TODO: continue making the code cleaner, be fast though!  
