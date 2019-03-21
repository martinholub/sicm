import matplotlib.pyplot as plt
import os

import numpy as np
import csv
import pandas as pd

from sicm.utils import utils

class DLS(Object):
    """Dynamic Light Scattering Measurmement

    Data musto iriginate from ZetaSizer, Malver.
    """
    def __init__(self):
        pass

    def _load_data(self):
        """tba"""

        # with open(tsvfile) as tf:
        #     reader = csv.DictReader(tf, dialect="excel-tab")
        #     for row in reader:

        cols = ["sizes", "intensities", "volumes", "numbers",
                "pdi", "z-average", ""]

        df = pd.read_csv(file_, sep="\t", encoding = "ISO-8859-1")
        samples = set(re.match("(.*) [1-3]$", x).group(1) for x in df["Sample Name"].values)
        columns = set(re.match("(.*?)(\[[0-9]+\]| ?)( ?\(.*\)| ?)$", x).group(1) for x in
                        df.columns.to_list()) 

        for sample in samples:
            df_sub = df[df["Sample Name"].str.match(sample + "[1-3]$")]
