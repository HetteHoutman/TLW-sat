import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from miscellaneous import check_argv_num

# TODO extend to multiple files? or ukv vs ukv_radsim etc.

check_argv_num(sys.argv, 4, '(results file 1, results file 1 name, results file 2, results file 2 name)')
file_1 = sys.argv[1]
filename_1 = file_1.split('/')[-1]
name_1 = sys.argv[2]

file_2 = sys.argv[3]
filename_2 = file_2.split('/')[-1]
name_2 = sys.argv[4]

df1 = pd.read_csv(file_1, index_col=[0, 1, 2], parse_dates=[0], dayfirst=True).sort_index()
df2 = pd.read_csv(file_2, index_col=[0, 1, 2], parse_dates=[0], dayfirst=True).sort_index()

df1.sort_index()
df2.sort_index()

# lambda
xy_line = [min(df1[['lambda', 'lambda_min', 'lambda_max']].min(axis=None), df2[['lambda', 'lambda_min', 'lambda_max']].min(axis=None)),
           max(df1[['lambda', 'lambda_min', 'lambda_max']].max(axis=None), df2[['lambda', 'lambda_min', 'lambda_max']].max(axis=None))]
plt.plot(xy_line, xy_line, 'k--', zorder=0)

for i, idx in enumerate(df1.index):
    xerr = [[df1.loc[idx, 'lambda'] - df1.loc[idx, 'lambda_min']],
            [df1.loc[idx, 'lambda_max'] - df1.loc[idx, 'lambda']]]
    yerr = [[df2.loc[idx, 'lambda'] - df2.loc[idx, 'lambda_min']],
            [df2.loc[idx, 'lambda_max'] - df2.loc[idx, 'lambda']]]
    plt.errorbar(df1.loc[idx, 'lambda'], df2.loc[idx, 'lambda'],
                 xerr=xerr,
                 yerr=yerr,
                 label=f'{idx[0].date()}_{idx[2]}h {idx[1]}',
                 capsize=3, marker='s', c=plt.cm.tab20(i))

plt.legend()
plt.title(f'{filename_1[:-12]} vs. {filename_2[:-12]}')
plt.xlabel(f'{name_1} wavelength (km)')
plt.ylabel(f'{name_2} wavelength (km)')
plt.tight_layout()
plt.savefig(f'plots/{filename_1[:-12]}_vs_{filename_2[:-12]}_lambda_comparison.png', dpi=300)
plt.show()

# theta
xy_line = [min(df1[['theta', 'theta_min', 'theta_max']].min(axis=None), df2[['theta', 'theta_min', 'theta_max']].min(axis=None)),
           max(df1[['theta', 'theta_min', 'theta_max']].max(axis=None), df2[['theta', 'theta_min', 'theta_max']].max(axis=None))]
plt.plot(xy_line, xy_line, 'k--', zorder=0)

for i, idx in enumerate(df1.index):
    xerr = [[df1.loc[idx, 'theta'] - df1.loc[idx, 'theta_min']],
            [df1.loc[idx, 'theta_max'] - df1.loc[idx, 'theta']]]
    yerr = [[df2.loc[idx, 'theta'] - df2.loc[idx, 'theta_min']],
            [df2.loc[idx, 'theta_max'] - df2.loc[idx, 'theta']]]
    plt.errorbar(df1.loc[idx, 'theta'], df2.loc[idx, 'theta'],
                 xerr=xerr,
                 yerr=yerr,
                 label=f'{idx[0].date()}_{idx[2]}h {idx[1]}',
                 capsize=3, marker='s', c=plt.cm.tab20(i))

plt.legend()
plt.title(f'{filename_1[:-12]} vs. {filename_2[:-12]}')
plt.xlabel(f'{name_1} wavevector direction from north (deg)')
plt.ylabel(f'{name_2} wavevector direction from north (deg)')
plt.tight_layout()
plt.savefig(f'plots/{filename_1[:-12]}_vs_{filename_2[:-12]}_theta_comparison.png', dpi=300)
plt.show()
