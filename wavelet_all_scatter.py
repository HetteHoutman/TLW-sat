import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
from miscellaneous import check_argv_num, k_spaced_lambda, create_bins_from_midpoints
from wavelet_plot import plot_result_lambda_hist

# prepare data
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

df = df1.merge(df2, how='inner', left_index=True, right_index=True, suffixes=('_' + name_1, '_' + name_2))
del (df1, df2)

l = df[[f'lambda_{name_1}', f'lambda_min_{name_1}', f'lambda_max_{name_1}',
        f'lambda_{name_2}', f'lambda_min_{name_2}', f'lambda_max_{name_2}']]

t = df[[f'theta_{name_1}', f'theta_min_{name_1}', f'theta_max_{name_1}',
        f'theta_{name_2}', f'theta_min_{name_2}', f'theta_max_{name_2}']]

# plot lambda
xy_line = [l.min(axis=None), l.max(axis=None)]
plt.plot(xy_line, xy_line, 'k--', zorder=0)

plt.errorbar(l[f'lambda_{name_1}'], l[f'lambda_{name_2}'],
             xerr=[l[f'lambda_{name_1}'] - l[f'lambda_min_{name_1}'],
                   l[f'lambda_max_{name_1}'] - l[f'lambda_{name_1}']],
             yerr=[l[f'lambda_{name_2}'] - l[f'lambda_min_{name_2}'],
                   l[f'lambda_max_{name_2}'] - l[f'lambda_{name_2}']],
             capsize=3, marker='s', c='k', linestyle='')

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

lambdas, lambdas_edges = k_spaced_lambda([5, 35], 40)
plot_result_lambda_hist(l[f'lambda_{name_1}'], l[f'lambda_{name_2}'], lambdas_edges,
                        label1=f'{name_1} wavelength (km)', label2=f'{name_2} wavelength (km)')
plt.plot(xy_line, xy_line, 'k--')
plt.tight_layout()
plt.savefig(f'plots/results/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_lambda_comparison_hist.png', dpi=300)
plt.show()

# plot theta
xy_line = [0, 180]
plt.plot(xy_line, xy_line, 'k--', zorder=0)

xerr = [(t[f'theta_{name_1}'] - t[f'theta_min_{name_1}']) % 180,
        (t[f'theta_max_{name_1}'] - t[f'theta_{name_1}']) % 180]
yerr = [(t[f'theta_{name_2}'] - t[f'theta_min_{name_2}']) % 180,
        (t[f'theta_max_{name_2}'] - t[f'theta_{name_2}']) % 180]

plt.errorbar(t[f'theta_{name_1}'], t[f'theta_{name_2}'],
             xerr=xerr, yerr=yerr, capsize=3, marker='s', c='k', linestyle='')

# overshoots
mask = t[f'theta_max_{name_1}'] < t[f'theta_{name_1}']
plt.errorbar(t[f'theta_{name_1}'][mask] - 180, t[f'theta_{name_2}'][mask],
             xerr=[xerr[0][mask], xerr[1][mask]], yerr=[yerr[0][mask], yerr[1][mask]],
             capsize=3, marker='s', c='k', linestyle='')

mask = t[f'theta_min_{name_1}'] > t[f'theta_{name_1}']
plt.errorbar(t[f'theta_{name_1}'][mask] + 180, t[f'theta_{name_2}'][mask],
             xerr=[xerr[0][mask], xerr[1][mask]], yerr=[yerr[0][mask], yerr[1][mask]],
             capsize=3, marker='s', c='k', linestyle='')

mask = t[f'theta_max_{name_2}'] < t[f'theta_{name_2}']
plt.errorbar(t[f'theta_{name_1}'][mask], t[f'theta_{name_2}'][mask] - 180,
             xerr=[xerr[0][mask], xerr[1][mask]], yerr=[yerr[0][mask], yerr[1][mask]],
             capsize=3, marker='s', c='k', linestyle='')

mask = t[f'theta_min_{name_2}'] > t[f'theta_{name_2}']
plt.errorbar(t[f'theta_{name_1}'][mask], t[f'theta_{name_2}'][mask] + 180,
             xerr=[xerr[0][mask], xerr[1][mask]], yerr=[yerr[0][mask], yerr[1][mask]],
             capsize=3, marker='s', c='k', linestyle='')

# plt.legend()
plt.title(f'{filename_1[:-4]} vs. {filename_2[:-4]}')
plt.xlabel(f'{name_1} wavevector direction from north (deg)')
plt.ylabel(f'{name_2} wavevector direction from north (deg)')
plt.xlim(0, 180)
plt.ylim(0, 180)
plt.tight_layout()
plt.savefig(f'plots/results/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_theta_comparison.png', dpi=300)
plt.show()

thetas = np.arange(0, 180, 5)
thetas_edges = create_bins_from_midpoints(thetas)

plt.hist2d(t[f'theta_{name_1}'], t[f'theta_{name_2}'], bins=[thetas_edges, thetas_edges])
plt.plot(xy_line, xy_line, 'k--')
plt.colorbar(label='Count')
plt.xlabel(f'{name_1} wavevector direction from north (deg)')
plt.ylabel(f'{name_2} wavevector direction from north (deg)')
plt.tight_layout()
plt.savefig(f'plots/results/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_theta_comparison_hist.png', dpi=300)
plt.show()