import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ukv800 = pd.read_csv('../tephiplot/wavelet_results/ukv_newalg_wind_800.csv', parse_dates=[0])
ukv700 = pd.read_csv('../tephiplot/wavelet_results/ukv_newalg_wind_700.csv', parse_dates=[0])
ukv600 = pd.read_csv('../tephiplot/wavelet_results/ukv_newalg_wind_600.csv', parse_dates=[0])
sat = pd.read_csv('../tephiplot/wavelet_results/sat_newalg_wind.csv', parse_dates=[0])

stds_600, stds_700, stds_800,stds_sat = [], [], [], []
for direc in glob.glob('/storage/silver/metstudent/phd/sw825517/ukv_data/2023*/*'):
    stds_600.append(np.load(direc + '/std_600.npy')[0])
    stds_700.append(np.load(direc + '/std.npy')[0])
    stds_800.append(np.load(direc + '/std_800.npy')[0])
    direc = direc.replace('ukv_data', 'sat_data')
    stds_sat.append(np.load(direc + '/std.npy')[0])

stds = np.array([stds_600, stds_700, stds_800, stds_sat]).mean(axis=1)

lambda_range = [3,35]
lambdas_edges = np.linspace(*lambda_range, 21)
labels = [rf'800hPa, {ukv800.shape[0]} wvs., $\bar{{\lambda}}$ = {ukv800['lambda'].mean():.02f} km, $\bar{{\sigma_w}}$ = {stds[2]:.03f} m/s',
          rf'700hPa, {ukv700.shape[0]} wvs., $\bar{{\lambda}}$ = {ukv700['lambda'].mean():.02f} km, $\bar{{\sigma_w}}$ = {stds[1]:.03f} m/s',
          rf'600hPa, {ukv600.shape[0]} wvs., $\bar{{\lambda}}$ = {ukv600['lambda'].mean():.02f} km, $\bar{{\sigma_w}}$ = {stds[0]:.03f} m/s']

colors = plt.cm.plasma([0, 0.5, 1])

plt.figure(figsize=(15, 5))
plt.hist(sat['lambda'], bins=lambdas_edges, label=rf'sat, {sat.shape[0]} wvs., $\bar{{\lambda}}$ = {sat['lambda'].mean():.02f} km',
         density=True, histtype='step', linestyle='--', hatch='//', color='k')
plt.hist([ukv800['lambda'], ukv700['lambda'], ukv600['lambda']], bins=lambdas_edges, label=labels, density=True, zorder=100, color=colors)

plt.legend()
plt.show()