import matplotlib.pyplot as plt
import numpy as np
import iris
import glob

diff_bins = np.arange(-90.625, 90, 1.25)
diff_vals = (diff_bins[1:] + diff_bins[:-1]) / 2
theta_bins = np.arange(-2.5, 180, 5)
theta_vals = (theta_bins[1:] + theta_bins[:-1])/2
# wind_bins = np.arange(-362.5, 365, 5)
wind_bins = theta_bins
# wind_vals = (wind_bins[1:] + wind_bins[:-1])/2
wind_vals = theta_vals

hist_diff_sat_800 = np.zeros(len(diff_vals))
hist_diff_sat_800_nwr = np.zeros(len(diff_vals))
hist_diff_ukv_800 = np.zeros(len(diff_vals))
hist_diff_ukv_800_nwr = np.zeros(len(diff_vals))
hist_diff_sat_700 = np.zeros(len(diff_vals))
hist_diff_sat_700_nwr = np.zeros(len(diff_vals))
diffs_800, diffs_800_nwr = [], []
weights_800, weights_800_nwr = [], []
diffs_700, diffs_700_nwr = [], []
weights_700, weights_700_nwr = [], []
diffs_ukv_800, diffs_ukv_800_nwr = [], []
weights_ukv_800, weights_ukv_800_nwr = [], []

hist_wind_theta_sat = np.zeros((len(theta_vals), len(wind_vals)))
hist_wind_theta_sat_nwr = np.zeros((len(theta_vals), len(wind_vals)))
hist_wind_theta_ukv = np.zeros((len(theta_vals), len(wind_vals)))
hist_wind_theta_ukv_nwr = np.zeros((len(theta_vals), len(wind_vals)))

for direc in glob.glob('/storage/silver/metstudent/phd/sw825517/sat_data/2023*/*'):
    try:
        wind_dir_800 = iris.load(direc + '/wind_dir_800.nc')[0]
        max_thetas_sat_800 = np.ma.masked_where(np.load(direc + '/mask_800.npy'), np.load(direc + '/max_thetas_800.npy'))
        max_thetas_nwr_sat_800 = np.ma.masked_where(np.load(direc + '/mask_nwr_800.npy'), np.load(direc + '/max_thetas_nwr_800.npy'))
        sat_diff_800 = max_thetas_sat_800 - wind_dir_800.data[::-1]
        sat_diff_800_nwr = max_thetas_nwr_sat_800 - wind_dir_800.data[::-1]

        wind_dir_700 = iris.load(direc + '/wind_dir.nc')[0]
        max_thetas_sat_700 = np.ma.masked_where(np.load(direc + '/mask.npy'), np.load(direc + '/max_thetas.npy'))
        max_thetas_nwr_sat_700 = np.ma.masked_where(np.load(direc + '/mask_nwr.npy'), np.load(direc + '/max_thetas_nwr.npy'))
        sat_diff_700 = max_thetas_sat_700 - wind_dir_700.data[::-1]
        sat_diff_700_nwr = max_thetas_nwr_sat_700 - wind_dir_700.data[::-1]

        direc = direc.replace('sat_data', 'ukv_data')
        max_thetas_ukv_800 = np.ma.masked_where(np.load(direc + '/mask_800.npy'), np.load(direc + '/max_thetas_800.npy'))
        max_thetas_nwr_ukv_800 = np.ma.masked_where(np.load(direc + '/mask_nwr_800.npy'), np.load(direc + '/max_thetas_nwr_800.npy'))
        ukv_diff_800 = max_thetas_ukv_800 - wind_dir_800.data[::-1]
        ukv_diff_800_nwr = max_thetas_nwr_ukv_800 - wind_dir_800.data[::-1]

        hist_diff_sat_800 += np.histogram(sat_diff_800[~sat_diff_800.mask], bins=diff_bins)[0]
        hist_diff_sat_800_nwr += np.histogram(sat_diff_800_nwr[~sat_diff_800_nwr.mask], bins=diff_bins)[0]

        hist_wind_theta_sat += np.histogram2d(max_thetas_sat_800[~max_thetas_sat_800.mask].data, wind_dir_800.data[::-1][~max_thetas_sat_800.mask].data % 180, bins=(theta_bins, wind_bins))[0]
        hist_wind_theta_sat_nwr += np.histogram2d(max_thetas_nwr_sat_800[~max_thetas_nwr_sat_800.mask].data, wind_dir_800.data[::-1][~max_thetas_nwr_sat_800.mask].data % 180, bins=(theta_bins, wind_bins))[0]
        hist_wind_theta_ukv += np.histogram2d(max_thetas_ukv_800[~max_thetas_ukv_800.mask].data, wind_dir_800.data[::-1][~max_thetas_ukv_800.mask].data % 180, bins=(theta_bins, wind_bins))[0]
        hist_wind_theta_ukv_nwr += np.histogram2d(max_thetas_nwr_ukv_800[~max_thetas_nwr_ukv_800.mask].data, wind_dir_800.data[::-1][~max_thetas_nwr_ukv_800.mask].data % 180, bins=(theta_bins, wind_bins))[0]

        if not np.isnan(sat_diff_800[~sat_diff_800.mask].data.mean()):
            diffs_800.append(sat_diff_800[~sat_diff_800.mask].data.mean())
            weights_800.append(sat_diff_800[~sat_diff_800.mask].size)
        if not np.isnan(sat_diff_800_nwr[~sat_diff_800_nwr.mask].data.mean()):
            diffs_800_nwr.append(sat_diff_800_nwr[~sat_diff_800_nwr.mask].data.mean())
            weights_800_nwr.append(sat_diff_800_nwr[~sat_diff_800_nwr.mask].size)

        hist_diff_sat_700 += np.histogram(sat_diff_700[~sat_diff_700.mask], bins=diff_bins)[0]
        hist_diff_sat_700_nwr += np.histogram(sat_diff_700_nwr[~sat_diff_700_nwr.mask], bins=diff_bins)[0]
        if not np.isnan(sat_diff_700[~sat_diff_700.mask].data.mean()):
            diffs_700.append(sat_diff_700[~sat_diff_700.mask].data.mean())
            weights_700.append(sat_diff_700[~sat_diff_700.mask].size)
        if not np.isnan(sat_diff_700_nwr[~sat_diff_700_nwr.mask].data.mean()):
            diffs_700_nwr.append(sat_diff_700_nwr[~sat_diff_700_nwr.mask].data.mean())
            weights_700_nwr.append(sat_diff_700_nwr[~sat_diff_700_nwr.mask].size)

        hist_diff_ukv_800 += np.histogram(ukv_diff_800[~ukv_diff_800.mask], bins=diff_bins)[0]
        hist_diff_ukv_800_nwr += np.histogram(ukv_diff_800_nwr[~ukv_diff_800_nwr.mask], bins=diff_bins)[0]
        if not np.isnan(ukv_diff_800[~ukv_diff_800.mask].data.mean()):
            diffs_ukv_800.append(ukv_diff_800[~ukv_diff_800.mask].data.mean())
            weights_ukv_800.append(ukv_diff_800[~ukv_diff_800.mask].size)
        if not np.isnan(ukv_diff_800_nwr[~ukv_diff_800_nwr.mask].data.mean()):
            diffs_ukv_800_nwr.append(ukv_diff_800_nwr[~ukv_diff_800_nwr.mask].data.mean())
            weights_ukv_800_nwr.append(ukv_diff_800_nwr[~ukv_diff_800_nwr.mask].size)
    except:
        pass

fig, ax = plt.subplots(2, 2, figsize=(9,6))
p1 = ax[0,0].pcolormesh(wind_bins, theta_bins, hist_wind_theta_sat)
fig.colorbar(p1)
ax[0,0].set_title(r'sat, $\Delta \leq $' + str(50) + rf'$\degree$')

p2 = ax[0,1].pcolormesh(wind_bins, theta_bins, hist_wind_theta_sat_nwr)
fig.colorbar(p2)
ax[0,1].set_title(r'sat, all $\vartheta$ allowed')

p3 = ax[1,0].pcolormesh(wind_bins, theta_bins, hist_wind_theta_ukv)
fig.colorbar(p3)
ax[1,0].set_title(r'UKV 800hPa, $\Delta \leq $' + str(50) + rf'$\degree$')

p4 = ax[1,1].pcolormesh(wind_bins, theta_bins, hist_wind_theta_ukv_nwr)
fig.colorbar(p4)
ax[1,1].set_title(r'UKV 800hPa, all $\vartheta$ allowed')

fig.supylabel(r'$\vartheta$ ($\degree$)')
fig.supxlabel(r'800hPa wind direction ($\degree$)')
for (m,n), subplot  in np.ndenumerate(ax):
    subplot.contour(wind_vals, theta_vals, wind_vals - theta_vals[:, np.newaxis],levels=[-130, -90, -50, 0, 50, 90, 130],
                    colors=['k', 'r', 'k', 'gray', 'k', 'r', 'k'], linestyles=['--']*7)
    subplot.set_xlim([0, 175])
    subplot.set_ylim([0, 175])

plt.show()

mean_diff_800, mean_diff_800_nwr = np.average(diffs_800, weights=weights_800), np.average(diffs_800_nwr, weights=weights_800_nwr)
mean_diff_ukv_800, mean_diff_ukv_800_nwr = np.average(diffs_ukv_800, weights=weights_ukv_800), np.average(diffs_ukv_800_nwr, weights=weights_ukv_800_nwr)
mean_diff_700, mean_diff_700_nwr = np.average(diffs_700, weights=weights_700), np.average(diffs_700_nwr, weights=weights_700_nwr)

plt.figure(figsize=(12, 5))
plt.bar(diff_vals, hist_diff_sat_800 / hist_diff_sat_800.sum() / 5, width=5, color='b', alpha=0.5, label=r'800hPa, $\vartheta \leq $' + str(50) + rf'$\degree$, mean diff: {mean_diff_800:0.1f}$\degree$')
plt.step(diff_vals, hist_diff_sat_800_nwr / hist_diff_sat_800_nwr.sum() / 5, linestyle='--', where='mid', color='gray', label=rf'800hPa, all $\vartheta$ allowed, mean diff: {mean_diff_800_nwr:0.1f}$\degree$')
plt.bar(diff_vals, hist_diff_sat_700 / hist_diff_sat_700.sum() / 5, width=5, color='r', alpha=0.5, label=r'700hPa, $\vartheta \leq $' + str(50) + rf'$\degree$, mean diff: {mean_diff_700:0.1f}$\degree$')
plt.step(diff_vals, hist_diff_sat_700_nwr / hist_diff_sat_700_nwr.sum() / 5, linestyle='--', where='mid', color='k', label=rf'700hPa, all $\vartheta$ allowed, mean diff: {mean_diff_700_nwr:0.1f}$\degree$')
plt.vlines(0, 0, (hist_diff_sat_800.max() / hist_diff_sat_800.sum() / 5), linestyle='--', color='k')
plt.legend()
plt.show()

plt.figure()
mean_diff_800, mean_diff_800_nwr = np.average(diffs_800, weights=weights_800), np.average(diffs_800_nwr, weights=weights_800_nwr)
mean_diff_700, mean_diff_700_nwr = np.average(diffs_700, weights=weights_700), np.average(diffs_700_nwr, weights=weights_700_nwr)
# plt.step(diff_vals, hist_diff_sat_800 / hist_diff_sat_800.sum() / 5, color='tab:blue', where='mid', label=r'Satellite, $\Delta \leq $' + str(50) + rf'$\degree$')
plt.step(diff_vals, hist_diff_sat_800_nwr / hist_diff_sat_800_nwr.sum() / 5, linestyle='-', where='mid', color='tab:blue', label=rf'Satellite')
# plt.step(diff_vals, hist_diff_ukv_800 / hist_diff_ukv_800.sum() / 5, color='tab:orange', where='mid', label=r'UKV 800hPa, $\Delta \leq $' + str(50) + rf'$\degree$')
plt.step(diff_vals, hist_diff_ukv_800_nwr / hist_diff_ukv_800_nwr.sum() / 5, linestyle='-', where='mid', color='tab:orange', label=rf'UKV 800 hPa')

# plt.vlines(0, 0, (hist_diff_sat_800.max() / hist_diff_sat_800.sum() / 5), linestyle='--', color='gray')
plt.vlines(mean_diff_800, 0, (hist_diff_sat_800.max() / hist_diff_sat_800.sum() / 5), linestyle='--', color='tab:blue')
plt.vlines(mean_diff_ukv_800, 0, (hist_diff_sat_800.max() / hist_diff_sat_800.sum() / 5), linestyle='--', color='tab:orange')
plt.vlines(-50, 0, (hist_diff_sat_800.max() / hist_diff_sat_800.sum() / 5), linestyle='-', color='k')
plt.vlines(50, 0, (hist_diff_sat_800.max() / hist_diff_sat_800.sum() / 5), linestyle='-', color='k')

plt.xlim(-90, 90)
plt.ylim(0, 0.004)
plt.xlabel(r'$\Delta (\degree)$')
plt.ylabel('Frequency')
plt.legend(frameon=True)
plt.savefig('plots/delta_plot.png', dpi=300)
plt.show()

print(r'Satellite, $\Delta \leq $' + str(50) + rf'$\degree$, mean diff: {mean_diff_800:0.1f}$\degree$')
print(rf'Satellite, all $\vartheta$ allowed, mean diff: {mean_diff_800_nwr:0.1f}$\degree$')
print(r'UKV 800hPa, $\Delta \leq $' + str(50) + rf'$\degree$, mean diff: {mean_diff_ukv_800:0.1f}$\degree$')
print(rf'UKV 800 hPa, all $\vartheta$ allowed, mean diff: {mean_diff_ukv_800_nwr:0.1f}$\degree$')