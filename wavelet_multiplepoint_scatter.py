import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from miscellaneous import log_spaced_lambda, create_bins_from_midpoints
from wavelet_plot import plot_result_lambda_hist

thetas = np.arange(0, 180, 5)
lambda_range = [3,35]

lambdas, lambdas_edges = log_spaced_lambda(lambda_range, (lambda_range[-1]/lambda_range[0]) ** (1 / (41 - 1)))
thetas_edges = create_bins_from_midpoints(thetas)

sat = pd.read_csv('../tephiplot/wavelet_results/sat_newalg_wind.csv', parse_dates=[0]).set_index(['date', 'region', 'h'])
ukv = pd.read_csv('../tephiplot/wavelet_results/ukv_newalg_wind_800.csv', parse_dates=[0]).set_index(['date', 'region', 'h'])

sat['kx'] = 2*np.pi / sat['lambda'] * np.cos(np.deg2rad(90 - sat['theta']))
sat['ky'] = 2*np.pi / sat['lambda'] * np.sin(np.deg2rad(90 - sat['theta']))

ukv['kx'] = 2*np.pi / ukv['lambda'] * np.cos(np.deg2rad(90 - ukv['theta']))
ukv['ky'] = 2*np.pi / ukv['lambda'] * np.sin(np.deg2rad(90 - ukv['theta']))

df = sat.copy()
#
# df['kx'] = 2 * np.pi / df['lambda'] * np.cos(np.deg2rad(90 - df.theta))
# df['ky'] = 2 * np.pi / df['lambda'] * np.sin(np.deg2rad(90 - df.theta))

for idx, row in df.iterrows():
    try:
        ukv_sub = ukv.loc[idx]
        dists = np.empty(len(ukv_sub))
        for i in range(len(ukv_sub)):
            dists[i] = np.linalg.norm(row[['kx', 'ky']].values - ukv.loc[idx][['kx', 'ky']].values[i])
            # dists[i] = abs(ukv.loc[idx]['lambda'].values[i] / row['lambda'] - 1)
        sel = np.argmin(dists)
        df.loc[idx, ['kx_ukv', 'ky_ukv']] = ukv_sub[['kx', 'ky']].values[sel]
    except KeyError:
        pass

df['lambda_ukv'] = 2*np.pi / np.sqrt(df['kx_ukv']**2 + df['ky_ukv']**2)
df['theta_ukv'] = 90 -  np.rad2deg(np.arctan2(df['ky_ukv'], df['kx_ukv']))

plot_result_lambda_hist(df['lambda'], df['lambda_ukv'], lambdas_edges,
                         label1='sat', label2='ukv')
q25 = df.groupby(pd.cut(df[f'lambda'], lambdas_edges, include_lowest=True, right=False))[f'lambda_ukv'].quantile(0.25)
medians = df.groupby(pd.cut(df[f'lambda'], lambdas_edges, include_lowest=True, right=False))[f'lambda_ukv'].quantile(0.5)
q75 = df.groupby(pd.cut(df[f'lambda'], lambdas_edges, include_lowest=True, right=False))[f'lambda_ukv'].quantile(0.75)
plt.plot(lambdas, medians, 'r-', label='UKV median')
plt.fill_between(lambdas, q25.values, q75.values, color='r', alpha=0.35, label='UKV 25th - 75th percentile')

plt.plot(lambda_range, lambda_range, 'k--', zorder=0)

plt.savefig('plots/l_test.png', dpi=300)
plt.close()


plt.hist2d(df['theta'], df['theta_ukv'], bins=[thetas_edges, thetas_edges], cmin=1)
plt.plot([0, 180], [0, 180], 'k--')
plt.savefig('plots/t_test.png', dpi=300)
plt.close()

