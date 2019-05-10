from .experiment import Experiment
from sicm.plots import plots

class CV(Experiment):
    def __init__(self, datadir, exp_name):
        super(CV, self).__init__(datadir, exp_name)
        # TODO: handle data, guessid params

    def plot(self, sel = None):
        x = [self._data["V1(V)"]]
        y = [self._data["Current1(A)"]*1e9]
        x_lab = [r"U [V]"]
        y_lab = [r"I [nA]"]
        leg = "Cyclic Voltamogram"
        fname = self.get_fpath()
        plots.plot_generic(x, y, x_lab, y_lab, leg, fname)


# import matplotlib.pyplot as plt
# import numpy as np
# import csv
#
# i_fname = "S:\\UsersData\\Martin\\2018\\12_Dec\\21\\cv_after_things_went_wrong_Current1 (A).tsv"
# v_fname = "S:\\UsersData\\Martin\\2018\\12_Dec\\21\\cv_after_things_went_wrong_V1 (V).tsv"
#
# def read_tsv(fname):
#     rows = []
#     with open(fname, "r") as rf:
#         tsvf = csv.reader(rf, delimiter = "\t")
#         for i, row in enumerate(tsvf):
#             try:
#                 row = np.asarray(list(map(float, filter(None, row))))
#             except ValueError as e:
#                 raise e
#         rows.append(row)
#     if len(rows) == 1: rows = rows[0]
#     return(rows)
#
# i = read_tsv(i_fname)
# v = read_tsv(v_fname)
#
# plt.style.use("seaborn")
# fig, ax = plt.subplots(figsize = (6.4, 4.8))
# ax.plot(v, i)
# ax.set_title("CV")
# ax.set_xlabel("v [V]")
# ax.set_ylabel("i [A]")
# plt.show()
