import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from miscellaneous import check_argv_num

# TODO extend to multiple files? or ukv vs ukv_radsim etc.

check_argv_num(sys.argv, 2, '(sat, ukv results fields)')
file_sat = sys.argv[1]
filename_sat = file_sat.split('/')[-1]
file_ukv = sys.argv[2]
filename_ukv = file_ukv.split('/')[-1]

sat_df = pd.read_csv(file_sat, index_col=[0, 1, 2]).sort_index()
ukv_df = pd.read_csv(file_ukv, index_col=[0, 1, 2]).sort_index()

# lambda
xy_line = [min(sat_df[['lambda', 'lambda_min', 'lambda_max']].min(axis=None), ukv_df[['lambda', 'lambda_min', 'lambda_max']].min(axis=None)),
           max(sat_df[['lambda', 'lambda_min', 'lambda_max']].max(axis=None), ukv_df[['lambda', 'lambda_min', 'lambda_max']].max(axis=None))]
plt.plot(xy_line, xy_line, 'k--', zorder=0)

for i, idx in enumerate(sat_df.index):
    xerr = [[sat_df.loc[idx, 'lambda'] - sat_df.loc[idx, 'lambda_min']],
            [sat_df.loc[idx, 'lambda_max'] - sat_df.loc[idx, 'lambda']]]
    yerr = [[ukv_df.loc[idx, 'lambda'] - ukv_df.loc[idx, 'lambda_min']],
            [ukv_df.loc[idx, 'lambda_max'] - ukv_df.loc[idx, 'lambda']]]
    plt.errorbar(sat_df.loc[idx, 'lambda'], ukv_df.loc[idx, 'lambda'],
                 xerr=xerr,
                 yerr=yerr,
                 label=f'{idx[0]}_{idx[2]}h {idx[1]}',
                 capsize=3, marker='s', c=plt.cm.tab20(i))

plt.legend()
plt.title(f'{filename_sat[:-12]} vs. {filename_ukv[:-12]}')
plt.xlabel('Satellite wavelength (km)')
plt.ylabel('UKV wavelength (km)')
plt.tight_layout()
plt.savefig(f'plots/{filename_sat[:-12]}_vs_{filename_ukv[:-12]}_lambda_comparison.png', dpi=300)
plt.show()

# theta
xy_line = [min(sat_df[['theta', 'theta_min', 'theta_max']].min(axis=None), ukv_df[['theta', 'theta_min', 'theta_max']].min(axis=None)),
           max(sat_df[['theta', 'theta_min', 'theta_max']].max(axis=None), ukv_df[['theta', 'theta_min', 'theta_max']].max(axis=None))]
plt.plot(xy_line, xy_line, 'k--', zorder=0)

for i, idx in enumerate(sat_df.index):
    xerr = [[sat_df.loc[idx, 'theta'] - sat_df.loc[idx, 'theta_min']],
            [sat_df.loc[idx, 'theta_max'] - sat_df.loc[idx, 'theta']]]
    yerr = [[ukv_df.loc[idx, 'theta'] - ukv_df.loc[idx, 'theta_min']],
            [ukv_df.loc[idx, 'theta_max'] - ukv_df.loc[idx, 'theta']]]
    plt.errorbar(sat_df.loc[idx, 'theta'], ukv_df.loc[idx, 'theta'],
                 xerr=xerr,
                 yerr=yerr,
                 label=f'{idx[0]}_{idx[2]}h {idx[1]}',
                 capsize=3, marker='s', c=plt.cm.tab20(i))

plt.legend()
plt.title(f'{filename_sat[:-12]} vs. {filename_ukv[:-12]}')
plt.xlabel('Satellite wavevector direction from north (deg)')
plt.ylabel('UKV wavevector direction from north (deg)')
plt.tight_layout()
plt.savefig(f'plots/{filename_sat[:-12]}_vs_{filename_ukv[:-12]}_theta_comparison.png', dpi=300)
plt.show()
