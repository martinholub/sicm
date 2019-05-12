from sicm.experiments.scan import Scan
import numpy as np
from sicm.utils.SubArgParser import SubArgParser

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
                scan_type)
    scan.plot_surface(args.plot_current, args.plot_slices, args.n_slices, args.center,
                        args.thickness, args.zrange, args.clip)

# elif args.what == "lockin":
#     # Data loc
#     # datadir = "S:/UsersData/Martin/2018/12_Dec/04/session_20181204_143956_07/sweep1_000"
#     datadir = os.path.join(datadir, exp_name)
#     file = "dev662_demods_0_sample_00000.csv"
#
#     data, date = io.load_data_lockin(datadir, file, chunk = 2)
#     plots.plot_lockin(data, date = date,
#         keys = list(zip(5*["frequency"], ["r", "phase", "phasepwr", "x", "y"])))
