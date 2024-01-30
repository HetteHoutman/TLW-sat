import os
import sys
import subprocess

import pandas as pd

df = pd.read_excel(sys.argv[1], index_col=[0,1])

for idx, row in df.iterrows():
    os.system(f"python fourier_analysis.py ../tephi_plot/settings/{idx[0][:4]}{idx[0][5:7]}{idx[0][8:10]}_{row.hour}.json {idx[1]}")
    # subprocess.call(["python", f"fourier_analysis.py",  f"../tephi_plot/settings/{idx[0][:4]}{idx[0][5:7]}{idx[0][8:10]}_{row.hour}.json",  f"{idx[1]}"], shell=True)