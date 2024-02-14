import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
from miscellaneous import check_argv_num

check_argv_num(sys.argv, 4, '(results file 1, results file 1 name, results file 2, results file 2 name)')
file_1 = sys.argv[1]
filename_1 = file_1.split('/')[-1]
name_1 = sys.argv[2]

file_2 = sys.argv[3]
filename_2 = file_2.split('/')[-1]
name_2 = sys.argv[4]

df1 = pd.read_csv(file_1, index_col=[0, 1, 2], parse_dates=[0]).sort_index()
df2 = pd.read_csv(file_2, index_col=[0, 1, 2], parse_dates=[0]).sort_index()

df1.sort_index()
df2.sort_index()

# lambda
xy_line = [min(df1[['lambda', 'lambda_min', 'lambda_max']].min(axis=None), df2[['lambda', 'lambda_min', 'lambda_max']].min(axis=None)),
           max(df1[['lambda', 'lambda_min', 'lambda_max']].max(axis=None), df2[['lambda', 'lambda_min', 'lambda_max']].max(axis=None))]
plt.plot(xy_line, xy_line, 'k--', zorder=0)

for i, idx in enumerate(df2.index):
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
plt.title(f'{filename_1[:-4]} vs. {filename_2[:-4]}')
plt.xlabel(f'{name_1} wavelength (km)')
plt.ylabel(f'{name_2} wavelength (km)')
plt.yscale('log')
plt.xscale('log')
ax = plt.gca()
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax.yaxis.set_minor_formatter(StrMethodFormatter('{x:.0f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.0f}'))
plt.tight_layout()
plt.savefig(f'plots/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_lambda_comparison.png', dpi=300)
plt.show()

# theta
xy_line = [0, 180]
plt.plot(xy_line, xy_line, 'k--', zorder=0)

overshoot1 = dict()
overshoot2 = dict()

for i, idx in enumerate(df2.index):
    xerr = [[(df1.loc[idx, 'theta'] - df1.loc[idx, 'theta_min']) % 180],
            [(df1.loc[idx, 'theta_max'] - df1.loc[idx, 'theta']) % 180]]
    yerr = [[(df2.loc[idx, 'theta'] - df2.loc[idx, 'theta_min']) % 180],
            [(df2.loc[idx, 'theta_max'] - df2.loc[idx, 'theta']) % 180]]

    plt.errorbar(df1.loc[idx, 'theta'], df2.loc[idx, 'theta'],
                 xerr=xerr,
                 yerr=yerr,
                 label=f'{idx[0].date()}_{idx[2]}h {idx[1]}',
                 capsize=3, marker='s', c=plt.cm.tab20(i))

    if df1.loc[idx, 'theta_max'] < df1.loc[idx, 'theta']:
        plt.errorbar(df1.loc[idx, 'theta'] - 180, df2.loc[idx, 'theta'],
                     xerr=xerr, yerr=yerr, capsize=3, marker='s', c=plt.cm.tab20(i))

    if df1.loc[idx, 'theta_min'] > df1.loc[idx, 'theta']:
        plt.errorbar(df1.loc[idx, 'theta'] + 180, df2.loc[idx, 'theta'],
                     xerr=xerr, yerr=yerr, capsize=3, marker='s', c=plt.cm.tab20(i))

    if df2.loc[idx, 'theta_max'] < df2.loc[idx, 'theta']:
        plt.errorbar(df1.loc[idx, 'theta'], df2.loc[idx, 'theta'] - 180,
                     xerr=xerr, yerr=yerr, capsize=3, marker='s', c=plt.cm.tab20(i))

    if df2.loc[idx, 'theta_min'] > df2.loc[idx, 'theta']:
        plt.errorbar(df1.loc[idx, 'theta'], df2.loc[idx, 'theta'] + 180,
                     xerr=xerr, yerr=yerr, capsize=3, marker='s', c=plt.cm.tab20(i))

plt.legend()
plt.title(f'{filename_1[:-4]} vs. {filename_2[:-4]}')
plt.xlabel(f'{name_1} wavevector direction from north (deg)')
plt.ylabel(f'{name_2} wavevector direction from north (deg)')
plt.xlim(0, 180)
plt.ylim(0, 180)
plt.tight_layout()
plt.savefig(f'plots/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_theta_comparison.png', dpi=300)
plt.show()
