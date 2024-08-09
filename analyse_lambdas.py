import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

try:
    leadtime = sys.argv[1]
    print(f'Setting leadtime to {leadtime}')
except IndexError:
    leadtime = 0
    print('No leadtime given, setting to 0')

ukv800 = pd.read_csv('../tephiplot/wavelet_results/ukv_newalg_wind_800.csv', parse_dates=[0])
ukv700 = pd.read_csv('../tephiplot/wavelet_results/ukv_newalg_wind.csv', parse_dates=[0])
ukv600 = pd.read_csv('../tephiplot/wavelet_results/ukv_newalg_wind_600.csv', parse_dates=[0])
sat = pd.read_csv('../tephiplot/wavelet_results/sat_newalg_wind_800.csv', parse_dates=[0])

lambda_range = [3,35]
lambdas_edges = np.linspace(*lambda_range, (lambda_range[1] - lambda_range[0])//2 + 1)
lambdas_vals = (lambdas_edges[1:] + lambdas_edges[:-1])/2

levels = ['600', '700', '800', 'sat']
suffixes = ['_600', '', '_800', '_800']
stds = {}
hist_lambdas = {}

if leadtime != 0:
    for i, suffix in enumerate(suffixes):
        suffixes[i] = f'_ld{leadtime}' + suffix

for level, suffix in zip(levels, suffixes):
    stds[level] = []
    hist_lambdas[level] = np.zeros(len(lambdas_vals))
    hist_lambdas[level + '_nwr'] = np.zeros(len(lambdas_vals))


for direc in glob.glob('/storage/silver/metstudent/phd/sw825517/ukv_data/2023*/*'):
    for level, suffix in zip(levels, suffixes):
        try:
            if level == 'sat':
                direc = direc.replace('ukv_data', 'sat_data')

            stds[level].append(np.load(direc + f'/std{suffix}.npy')[0])

            max_lambdas = np.ma.masked_where(np.load(direc + f'/mask{suffix}.npy'), np.load(direc + f'/max_lambdas{suffix}.npy'))
            max_lambdas_nwr = np.ma.masked_where(np.load(direc + f'/mask_nwr{suffix}.npy'), np.load(direc + f'/max_lambdas_nwr{suffix}.npy'))

            hist_lambdas[level] += np.histogram(max_lambdas[~max_lambdas.mask], bins=lambdas_edges)[0]
            hist_lambdas[level + '_nwr'] += np.histogram(max_lambdas_nwr[~max_lambdas_nwr.mask], bins=lambdas_edges)[0]

            if level == 'sat':
                direc = direc.replace('sat_data', 'ukv_data')
        except FileNotFoundError as error:
            print(error)
        except EOFError as error:
            print(error)
        #
        except ValueError as error:
            print(error)
avg_stds = np.array(tuple(stds.values())).mean(axis=1)

labels = [rf'UKV 800hPa, {ukv800.shape[0]} wvs., $\bar{{\lambda}}$ = {ukv800['lambda'].mean():.02f} km, $\bar{{\sigma_w}}$ = {avg_stds[2]:.03f} m/s',
          rf'UKV 700hPa, {ukv700.shape[0]} wvs., $\bar{{\lambda}}$ = {ukv700['lambda'].mean():.02f} km, $\bar{{\sigma_w}}$ = {avg_stds[1]:.03f} m/s',
          rf'UKV 600hPa, {ukv600.shape[0]} wvs., $\bar{{\lambda}}$ = {ukv600['lambda'].mean():.02f} km, $\bar{{\sigma_w}}$ = {avg_stds[0]:.03f} m/s']

colors = plt.cm.plasma([0, 0.5, 1])

plt.figure(figsize=(15, 5))
plt.hist(sat['lambda'], bins=lambdas_edges, label=rf'Satellite, {sat.shape[0]} wvs., $\bar{{\lambda}}$ = {sat['lambda'].mean():.02f} km',
         density=True, histtype='step', linestyle='--', hatch='//', color='k')
plt.hist([ukv800['lambda'],
          ukv700['lambda'], ukv600['lambda']], bins=lambdas_edges, label=labels, density=True, zorder=100, color=colors)
plt.legend()
plt.xticks(np.arange(3, 35 + 1, 2))
plt.xlabel('Wavelength (km)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('plots/height_selection.png', dpi=300)
plt.show()

# plt.figure(figsize=(15, 5))
# width = lambdas_edges[1] - lambdas_edges[0]
# for i, level in enumerate(levels[:-1]):
#     plt.bar(lambdas_vals - width*(-1 + i)/4, hist_lambdas[level] / hist_lambdas[level].sum() / width, label=level, zorder=100, color=colors[i], width=width/4)
# plt.bar(lambdas_vals, hist_lambdas['sat'] / hist_lambdas['sat'].sum() / width, label=rf'sat',
#         linestyle='--', hatch='//', color='k', width=width, fill=False)
# plt.legend()
# plt.title('Pixelwise')
# plt.show()