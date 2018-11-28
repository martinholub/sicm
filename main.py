from sicm import io, plots
import numpy as np

# Globals
datadir = "C:/Users/mholub/data/sicm/2018/11_Nov/27"

# Locals
exp_name = "scan1"

# Get files
files, date = io.get_files(datadir, exp_name)
result_ = io.load_result(files, exp_name)

# Select Line Number
uniqs, cnts = np.unique(result_["LineNumber"], return_counts=True)
linenos = np.arange(5, max(uniqs), 3)
result = io.downsample_to_linenumber(result_, linenos, which = "last")

plots.plot_surface(result)
