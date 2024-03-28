import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

df1 = pd.read_csv(sys.argv[1], parse_dates=[0])
df2 = pd.read_csv(sys.argv[2], parse_dates=[0])

for i, (df1idx, df1row) in enumerate(df1.iterrows()):
        for j, (df2idx, df2row) in enumerate(df2[(df2.date==df1row.date) & (df2.region==df1row.region) & (df2.h==df1row.h)].iterrows()):
            dist = None