import os
import sys

import numpy as np
import pandas as pd

df = pd.read_excel(sys.argv[1], index_col=[0, 1, 2], parse_dates=[0])

for idx, row in df.iterrows():
    if idx[2] is not np.nan:
        regions = idx[2].split('/')
        hour = int(idx[1])

        for region in regions:
            # only 2023 for now
            if idx[0].year == 2023 and region != 'none':
                # only run if not already run
                dir_name = f'./plots/{idx[0].date()}_{hour:02d}/{region}/'
                if os.path.isdir(dir_name):
                    if not os.listdir(dir_name):
                        print(f"Directory {dir_name} is empty")
                        os.system(f"python wavelet_analysis.py {idx[0].date()}_{hour:02d} {region}")
                    else:
                        pass
                else:
                    print(f"Directory {dir_name} does not exist")
                    os.system(f"python wavelet_analysis.py {idx[0].date()}_{hour:02d} {region}")

# for idx, row in df.iterrows():
#     if row.selected == 'x':
#         regions = idx[2].split('/')
#         for region in regions:
#             os.system(
#                 f"python wavelet_analysis.py {idx[0].strftime('%Y-%m-%d')}_{int(idx[1])} {region}")
        #
        # print(f"python wavelet_analysis.py {idx[0].date()}_{idx[1]} {regions[0]}")
        #