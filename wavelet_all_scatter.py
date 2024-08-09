import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
from miscellaneous import check_argv_num, k_spaced_lambda, create_bins_from_midpoints, log_spaced_lambda
from wavelet_plot import plot_result_lambda_hist, plot_polar_pcolormesh
from wavelet import angle_error

lambda_range = [3, 35]
n_lambda = 60

# prepare data
check_argv_num(sys.argv, 4, '(results file 1, results file 1 name, results file 2, results file 2 name)')
file_1 = sys.argv[1]
filename_1 = file_1.split('/')[-1]
name_1 = sys.argv[2]

file_2 = sys.argv[3]
filename_2 = file_2.split('/')[-1]
name_2 = sys.argv[4]

df1 = pd.read_csv(file_1, parse_dates=[0]).sort_index()
df2 = pd.read_csv(file_2, parse_dates=[0]).sort_index()

df1_orig = df1.copy()
df2_orig = df2.copy()

df1.drop_duplicates(['date', 'region', 'h'], inplace=True)
df2.drop_duplicates(['date', 'region', 'h'], inplace=True)

df1.set_index(['date', 'region', 'h'], inplace=True)
df2.set_index(['date', 'region', 'h'], inplace=True)

df1['lambda_min'] = df1['lambda'] - 0.01
df1['lambda_max'] = df1['lambda'] + 0.01
df1['theta_min'] = df1['theta'] - 0.01
df1['theta_max'] = df1['theta'] + 0.01

df2['lambda_min'] = df2['lambda'] - 0.01
df2['lambda_max'] = df2['lambda'] + 0.01
df2['theta_min'] = df2['theta'] - 0.01
df2['theta_max'] = df2['theta'] + 0.01

df = df1.merge(df2, how='inner', left_index=True, right_index=True, suffixes=('_' + name_1, '_' + name_2))

# df = df[(df['theta_ukv'] > 120) | (df['theta_ukv'] < 60)]
# df = df[df[f'lambda_{name_2}'] < 10]


df['lambda_diff'] = df[f'lambda_{name_1}'] - df[f'lambda_{name_2}']
df['theta_diff'] = angle_error(df[f'theta_{name_1}'], df[f'theta_{name_2}'])

l = df[[f'lambda_{name_1}', f'lambda_min_{name_1}', f'lambda_max_{name_1}',
        f'lambda_{name_2}', f'lambda_min_{name_2}', f'lambda_max_{name_2}']]

t = df[[f'theta_{name_1}', f'theta_min_{name_1}', f'theta_max_{name_1}',
        f'theta_{name_2}', f'theta_min_{name_2}', f'theta_max_{name_2}']]

# plot lambda
xy_line = lambda_range
plt.plot(xy_line, xy_line, 'k--', zorder=0)

for i, index in enumerate(df.index):
    c = 'k' if abs(df.theta_diff.loc[index]) <= 45 else 'r'
    plt.errorbar(l[f'lambda_{name_1}'].loc[index], l[f'lambda_{name_2}'].loc[index],
                 xerr=[[l[f'lambda_{name_1}'].loc[index] - l[f'lambda_min_{name_1}'].loc[index]],
                       [l[f'lambda_max_{name_1}'].loc[index] - l[f'lambda_{name_1}'].loc[index]]],
                 yerr=[[l[f'lambda_{name_2}'].loc[index] - l[f'lambda_min_{name_2}'].loc[index]],
                       [l[f'lambda_max_{name_2}'].loc[index] - l[f'lambda_{name_2}'].loc[index]]],
                 capsize=3, marker='s', color=c, linestyle='', alpha=0.5)

# plt.scatter(l[f'lambda_{name_1}'], l[f'lambda_{name_2}'], c=df.theta_diff, cmap='plasma', marker='s', alpha=0.5, vmin=0, vmax=90)
# plt.colorbar()

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
plt.savefig(f'plots/results/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_lambda_comparison.png', dpi=300)
plt.close()

lambdas, lambdas_edges = log_spaced_lambda(lambda_range, (lambda_range[-1]/lambda_range[0]) ** (1 / (41 - 1)))
plot_result_lambda_hist(l[f'lambda_{name_1}'], l[f'lambda_{name_2}'], lambdas_edges,
                        label1=f'{name_1} wavelength (km)', label2=f'{name_2} wavelength (km)')
plt.plot(xy_line, xy_line, 'k--', linewidth=2)
# q25 = df.groupby(pd.cut(df[f'lambda_{name_1}'], lambdas_edges, include_lowest=True, right=False))[f'lambda_{name_2}'].quantile(0.25)
medians = df.groupby(pd.cut(df[f'lambda_{name_1}'], lambdas_edges, include_lowest=True, right=False))[f'lambda_{name_2}'].quantile(0.5)
# q75 = df.groupby(pd.cut(df[f'lambda_{name_1}'], lambdas_edges, include_lowest=True, right=False))[f'lambda_{name_2}'].quantile(0.75)
plt.plot(lambdas, medians, 'r-', label='UKV median', linewidth=2)
# plt.fill_between(lambdas, q25.values, q75.values, color='r', alpha=0.35, label='UKV 25th - 75th percentile')
plt.annotate(rf'$\rho=${df[[f'lambda_{name_1}', f'lambda_{name_2}']].corr(method='spearman').iloc[0,1]:0.2f}'
             f'\n $N=${df.shape[0]}',
             xy=(32, 3.2), xycoords='data', ha='right')
plt.legend()
plt.xlim(lambda_range)
plt.ylim(lambda_range)
plt.tight_layout()
plt.savefig(f'plots/results/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_lambda_comparison_hist.png', dpi=300)
plt.close()

# plot theta
xy_line = [0, 180]
plt.plot(xy_line, xy_line, 'k--', zorder=0, linewidth=2)

xerr = np.array([(t[f'theta_{name_1}'] - t[f'theta_min_{name_1}']).values % 180,
        (t[f'theta_max_{name_1}'] - t[f'theta_{name_1}']).values % 180])
yerr = np.array([(t[f'theta_{name_2}'] - t[f'theta_min_{name_2}']).values % 180,
        (t[f'theta_max_{name_2}'] - t[f'theta_{name_2}']).values % 180])

for i, index in enumerate(df.index):
    plt.errorbar(t[f'theta_{name_1}'].loc[index], t[f'theta_{name_2}'].loc[index],
                 xerr=xerr[:, i].reshape(2,1), yerr=yerr[:, i].reshape(2,1), capsize=3, marker='s',
                 color=plt.cm.RdBu((df.lambda_diff.loc[index] - df.lambda_diff.mean())/np.abs(df.lambda_diff.max()) ), linestyle='', alpha=0.3)

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
# plt.colorbar()
plt.title(f'{filename_1[:-4]} vs. {filename_2[:-4]}')
plt.xlabel(f'{name_1} wavevector direction from north (deg)')
plt.ylabel(f'{name_2} wavevector direction from north (deg)')
plt.xlim(0, 180)
plt.ylim(0, 180)
plt.tight_layout()
plt.savefig(f'plots/results/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_theta_comparison.png', dpi=300)
plt.close()

thetas = np.arange(0, 180, 5)
thetas_edges = create_bins_from_midpoints(thetas)

plt.hist2d(t[f'theta_{name_1}'], t[f'theta_{name_2}'], bins=[thetas_edges, thetas_edges], cmin=1)
plt.plot(xy_line, xy_line, 'k--')
plt.colorbar(label='Number of cases')
plt.xlabel(rf'{name_1} orientation ($\degree$N)')
plt.ylabel(rf'{name_2} orientation ($\degree$N)')
plt.annotate(rf'$\rho=${df[[f'theta_{name_1}', f'theta_{name_2}']].corr(method='spearman').iloc[0,1]:0.2f}'
             f'\n $N=${df.shape[0]}',
             xy=(165, 5), xycoords='data', ha='right')
plt.tight_layout()
plt.savefig(f'plots/results/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_theta_comparison_hist.png', dpi=300)
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
lambdas, lambdas_edges = log_spaced_lambda(lambda_range, (lambda_range[-1]/lambda_range[0]) ** (1 / (41 - 1)))
plt.sca(ax1)
plot_result_lambda_hist(l[f'lambda_{name_1}'], l[f'lambda_{name_2}'], lambdas_edges,
                        label1=f'{name_1} wavelength (km)', label2=f'{name_2} wavelength (km)')
plt.plot(xy_line, xy_line, 'k--', linewidth=2)
# q25 = df.groupby(pd.cut(df[f'lambda_{name_1}'], lambdas_edges, include_lowest=True, right=False))[f'lambda_{name_2}'].quantile(0.25)
medians = df.groupby(pd.cut(df[f'lambda_{name_1}'], lambdas_edges, include_lowest=True, right=False))[f'lambda_{name_2}'].quantile(0.5)
# q75 = df.groupby(pd.cut(df[f'lambda_{name_1}'], lambdas_edges, include_lowest=True, right=False))[f'lambda_{name_2}'].quantile(0.75)
plt.plot(lambdas, medians, 'r-', label='UKV median', linewidth=2)
# plt.fill_between(lambdas, q25.values, q75.values, color='r', alpha=0.35, label='UKV 25th - 75th percentile')
plt.annotate(rf'$\rho=${df[[f'lambda_{name_1}', f'lambda_{name_2}']].corr(method='spearman').iloc[0,1]:0.2f}'
             f'\n $N=${df.shape[0]}',
             xy=(32, 3.2), xycoords='data', ha='right')
plt.legend()
plt.xlim(lambda_range)
plt.ylim(lambda_range)

plt.sca(ax2)
xy_line = [0,180]
plt.hist2d(t[f'theta_{name_1}'], t[f'theta_{name_2}'], bins=[thetas_edges, thetas_edges], cmin=1)
plt.plot(xy_line, xy_line, 'k--')
plt.colorbar(label='Number of cases')
plt.xlabel(rf'{name_1} orientation ($\degree$N)')
plt.ylabel(rf'{name_2} orientation ($\degree$N)')
plt.annotate(rf'$\rho=${df[[f'theta_{name_1}', f'theta_{name_2}']].corr(method='spearman').iloc[0,1]:0.2f}'
             f'\n $N=${df.shape[0]}',
             xy=(165, 5), xycoords='data', ha='right')
plt.tight_layout()
plt.savefig(f'plots/results/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_combined_hist.png', dpi=300)
plt.close()


#df.reset_index()['date'].dt.month.value_counts().sort_index().plot(kind='bar')
# df.reset_index()['region'].value_counts().plot(kind='bar')

# plt.scatter(df['lambda_diff']/df[f'lambda_{name_1}'], df['theta_diff'])
# plt.xlabel(f'(lambda_{name_1} - lambda_{name_2}) / lambda_{name_1}')
# plt.ylabel(f'theta_{name_1} - theta_{name_2} (deg)')
# plt.tight_layout()
# plt.close()
#
# plt.hist2d(df['lambda_diff']/df[f'lambda_{name_1}'], df['theta_diff'], bins=[40, 36])
# plt.xlabel(f'(lambda_{name_1} - lambda_{name_2}) / lambda_{name_1}')
# plt.ylabel(f'theta_{name_1} - theta_{name_2} (deg)')
# plt.tight_layout()
# plt.close()
#
# plt.hist2d(abs(df['lambda_diff']/df[f'lambda_{name_1}']), abs(df['theta_diff']), bins=[40, 18])
# plt.xlabel(f'(lambda_{name_1} - lambda_{name_2}) / lambda_{name_1}')
# plt.ylabel(f'theta_{name_1} - theta_{name_2} (deg)')
# plt.tight_layout()
# plt.close()
#
# plt.hist2d(df['lambda_diff']/df[f'lambda_{name_1}'], abs(df['theta_diff']), bins=[40, 18])
# plt.xlabel(f'(lambda_{name_1} - lambda_{name_2}) / lambda_{name_1}')
# plt.ylabel(f'theta_{name_1} - theta_{name_2} (deg)')
# plt.tight_layout()
# plt.close()

lambdas_edges = np.linspace(*lambda_range, 21)

plt.hist(df1_orig[f'lambda'], bins=lambdas_edges, label=name_1, density=True)
plt.hist(df2_orig[f'lambda'], hatch='/', fill=False, bins=lambdas_edges, label=name_2, density=True)
plt.xlabel('Wavelength (km)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(f'plots/results/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_lambda_hist.png', dpi=300)
plt.close()

plt.hist(df2_orig[f'theta'], bins=thetas_edges, label=name_2, density=True)
plt.hist(df1_orig[f'theta'], hatch='/', fill=False,  bins=thetas_edges, label=name_1, density=True)
plt.xlabel(r'Orientation ($\degree$N)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(f'plots/results/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_theta_hist.png', dpi=300)
plt.close()

lambdas, lambdas_edges = log_spaced_lambda(lambda_range, (lambda_range[-1]/lambda_range[0]) ** (1 / (21 - 1)))
hist_1 = np.histogram2d(df1_orig[f'lambda'], df1_orig[f'theta'], bins=[lambdas_edges, thetas_edges])
hist_1[0][hist_1[0]==0] = np.nan
hist_2 = np.histogram2d(df2_orig[f'lambda'], df2_orig[f'theta'], bins=[lambdas_edges, thetas_edges])
hist_2[0][hist_2[0] == 0] = np.nan

plot_polar_pcolormesh(*hist_1, cbarlabel='Number of cases', )
plt.annotate('a)', (0.15, 0.95), xycoords='figure fraction', weight='bold', fontsize=15)
plt.savefig(f'plots/results/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_{name_1}_2dhist.png', bbox_inches='tight', dpi=300)
plt.close()

plot_polar_pcolormesh(*hist_2, cbarlabel='Number of cases', )
plt.annotate('b)', (0.15, 0.95), xycoords='figure fraction', weight='bold', fontsize=15)
plt.savefig(f'plots/results/wavelet_{filename_1[:-4]}_vs_{filename_2[:-4]}_{name_2}_2dhist.png', bbox_inches='tight', dpi=300)
plt.close()