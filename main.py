from sicm import io, plots
import numpy as np
from sicm.utils.SubArgParser import SubArgParser

# TODO: Add also other args to parser, but keep flexible enough

ap = SubArgParser(prog="Cell line Sheet Replacement", add_help=True)
args = ap.parse_args()

if args.what == "scan":
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

elif args.what == "lockin":
    # Data loc
    datadir = "C:/Users/mholub/data/sicm/2018/11_Nov/30/session_20181130_145223_03/sweep1_001"

    file = "dev662_demods_0_sample_00000.csv"
    data, date = io.load_data_lockin(datadir, file, chunk = 3)
    plots.plot_lockin(data, date = date,
        keys = list(zip(5*["frequency"], ["r", "phase", "phasepwr", "x", "y"])))
