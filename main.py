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
    scan = Scan(data_dir, exp_name, args.do_correct, args.yrange, args.xrange,
                args.is_constant_distance)
    scan.plot_surface(args.plot_current)

# elif args.what == "lockin":
#     # Data loc
#     # datadir = "S:/UsersData/Martin/2018/12_Dec/04/session_20181204_143956_07/sweep1_000"
#     datadir = os.path.join(datadir, exp_name)
#     file = "dev662_demods_0_sample_00000.csv"
#
#     data, date = io.load_data_lockin(datadir, file, chunk = 2)
#     plots.plot_lockin(data, date = date,
#         keys = list(zip(5*["frequency"], ["r", "phase", "phasepwr", "x", "y"])))
