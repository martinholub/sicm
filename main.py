from sicm.experiments.scan import Scan
import numpy as np
from sicm.utils.SubArgParser import SubArgParser
import matplotlib as mpl
from sicm.plots.plots import _set_rcparams

ap = SubArgParser(prog="SICM Toolkit", add_help=True)
args = ap.parse_args()
# Pass changing params on CL
data_dir = args.datadir
exp_name = args.exp_name
if args.what == "scan":
    # You should run this from command line for interactive plotting to work properly
    scan_type = "scan"
    if args.is_constant_distance:
        scan_type = "constant_distance"
    scan = Scan(data_dir, exp_name, args.yrange, args.xrange, args.do_correct,
                args.convert, scan_type)
    _set_rcparams(style = args.plot_style)
    scan.plot_surface(args.plot_current, args.plot_slices, args.n_slices, args.center,
                        args.thickness, args.zrange, args.clip, args.scale,
                        args.n_levels, args.descriptor, args.overlay, args.adjust)

# elif args.what == "lockin":
#     # Data loc
#     # datadir = "S:/UsersData/Martin/2018/12_Dec/04/session_20181204_143956_07/sweep1_000"
#     datadir = os.path.join(datadir, exp_name)
#     file = "dev662_demods_0_sample_00000.csv"
#
#     data, date = io.load_data_lockin(datadir, file, chunk = 2)
#     plots.plot_lockin(data, date = date,
#         keys = list(zip(5*["frequency"], ["r", "phase", "phasepwr", "x", "y"])))
