from sicm import io, plots
import numpy as np
from sicm.utils.SubArgParser import SubArgParser

# TODO: Add also other args to parser, but keep flexible enough

ap = SubArgParser(prog="SICM Toolkit", add_help=True)
args = ap.parse_args()

if args.what == "scan":
    # Globals
    datadir = "C:/Users/mholub/data/sicm/2018/11_Nov/26"
    datadir = "S:/UsersData/Martin/2018/12_Dec/12/sicm/exp4"
    # Locals
    exp_name = "scan3"
    exp_name = "scan_hopping_qtf_current_veryfast"

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
    datadir = "S:/UsersData/Martin/2018/12_Dec/04/session_20181204_143956_07/sweep1_000"

    file = "dev662_demods_0_sample_00000.csv"
    data, date = io.load_data_lockin(datadir, file, chunk = 2)
    plots.plot_lockin(data, date = date,
        keys = list(zip(5*["frequency"], ["r", "phase", "phasepwr", "x", "y"])))
