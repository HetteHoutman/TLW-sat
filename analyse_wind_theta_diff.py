import matplotlib.pyplot as plt
import numpy as np
import iris
import glob

diff_bins = np.arange(-92.5, 90, 5)
diff_vals = (diff_bins[1:] + diff_bins[:-1]) / 2
wind_bins = np.arange(-362.5, 360, 5)
wind_vals = (wind_bins[1:] + wind_bins[:-1])/2
theta_bins = np.arange(-2.5, 180, 5)
theta_vals = (theta_bins[1:] + theta_bins[:-1])/2

hist_diff_600 = np.zeros(len(diff_vals))
hist_diff_700 = np.zeros(len(diff_vals))
hist_diff_800 = np.zeros(len(diff_vals))
hist_wind_600 = np.zeros(len(wind_vals))
hist_wind_700 = np.zeros(len(wind_vals))
hist_wind_800 = np.zeros(len(wind_vals))
hist_theta_600 = np.zeros(len(theta_vals))
hist_theta_700 = np.zeros(len(theta_vals))
hist_theta_800 = np.zeros(len(theta_vals))
for direc in glob.glob('/storage/silver/metstudent/phd/sw825517/ukv_data/2023*/*'):
    max_thetas_600 = np.ma.masked_where(np.load(direc + '/mask_600.npy'), np.load(direc + '/max_thetas_600.npy'))
    wind_dir_600 = iris.load(direc + '/wind_dir_600.nc')[0]
    diff_600 = max_thetas_600 - wind_dir_600.data[::-1]
    hist_diff_600 += np.histogram(diff_600[~diff_600.mask], bins=diff_bins)[0]
    hist_wind_600 += np.histogram(wind_dir_600.data.flatten(), bins=wind_bins)[0]
    hist_theta_600 += np.histogram(max_thetas_600[~max_thetas_600.mask], bins=theta_bins)[0]

    max_thetas_700 = np.ma.masked_where(np.load(direc + '/mask.npy'), np.load(direc + '/max_thetas.npy'))
    wind_dir_700 = iris.load(direc + '/wind_dir.nc')[0]
    diff_700 = max_thetas_700 - wind_dir_700.data[::-1]
    hist_diff_700 += np.histogram(diff_700[~diff_700.mask], bins=diff_bins)[0]
    hist_wind_700 += np.histogram(wind_dir_700.data.flatten(), bins=wind_bins)[0]
    hist_theta_700 += np.histogram(max_thetas_700[~max_thetas_700.mask], bins=theta_bins)[0]

    max_thetas_800 = np.ma.masked_where(np.load(direc + '/mask_800.npy'), np.load(direc + '/max_thetas_800.npy'))
    wind_dir_800 = iris.load(direc + '/wind_dir_800.nc')[0]
    diff_800 = max_thetas_800 - wind_dir_800.data[::-1]
    hist_diff_800 += np.histogram(diff_800[~diff_800.mask], bins=diff_bins)[0]
    hist_wind_800 += np.histogram(wind_dir_800.data.flatten(), bins=wind_bins)[0]
    hist_theta_800 += np.histogram(max_thetas_800[~max_thetas_800.mask], bins=theta_bins)[0]

hist_theta_sat = np.zeros(len(theta_vals))
hist_diff_sat = np.zeros(len(diff_vals))
hist_diff_nwr_sat = np.zeros(len(diff_vals))
sat_diff_dic = {}
for label in ['600', '700', '800']:
    sat_diff_dic[label] = np.zeros(len(diff_vals))
    sat_diff_dic[label + '_nwr'] = np.zeros(len(diff_vals))
for direc in glob.glob('/storage/silver/metstudent/phd/sw825517/sat_data/2023*/*'):
    wind_dir_sat = iris.load(direc + '/wind_dir.nc')[0]
    wind_dir_800 = iris.load(direc.replace('sat_data', 'ukv_data') + '/wind_dir_800.nc')[0]
    wind_dir_600 = iris.load(direc.replace('sat_data', 'ukv_data') + '/wind_dir_600.nc')[0]
    max_thetas_sat = np.ma.masked_where(np.load(direc + '/mask.npy'), np.load(direc + '/max_thetas.npy'))
    max_thetas_nwr_sat = np.ma.masked_where(np.load(direc + '/mask_nwr.npy'), np.load(direc + '/max_thetas_nwr.npy'))
    hist_theta_sat += np.histogram(max_thetas_sat[~max_thetas_sat.mask], bins=theta_bins)[0]

    for wind_dir, label in zip([wind_dir_600, wind_dir_sat, wind_dir_800], ['600', '700', '800']):
        sat_diff = max_thetas_sat - wind_dir.data[::-1]
        sat_diff_nwr = max_thetas_nwr_sat - wind_dir.data[::-1]
        sat_diff_dic[label] += np.histogram(sat_diff[~sat_diff.mask], bins=diff_bins)[0]
        sat_diff_dic[label + '_nwr'] += np.histogram(sat_diff_nwr[~sat_diff_nwr.mask], bins=diff_bins)[0]


diff_hists = [sat_diff_dic['700'], sat_diff_dic['700_nwr'], hist_diff_800, hist_diff_700, hist_diff_600]
diff_labels = ['sat', 'sat_nwr', '800', '700', '600']
colors = ['k', 'gray', 'b', 'g', 'r']
for hist, label, color in zip(diff_hists, diff_labels, colors):
    if label == 'sat' or label == 'sat_nwr':
        plt.step(diff_vals, hist / hist.sum(), where='mid', label=label, color=color, linestyle='--', zorder=100)
    else:
        plt.bar(diff_vals, hist / hist.sum(), label=label, color=color, width=5, alpha=0.5)
plt.vlines(0, 0, (np.array(diff_hists).T / np.array(diff_hists).sum(1)).max(), 'k', '-')
plt.legend()
plt.title('Orientation - wind direction')
plt.show()

theta_hists = [hist_theta_600, hist_theta_700, hist_theta_800, hist_theta_sat][::-1]
theta_labels = ['600', '700', '800', 'sat'][::-1]
colors = ['k', 'b', 'g', 'r']
for hist, label, color in zip(theta_hists, theta_labels, colors):
     if label == 'sat':
         plt.step(theta_vals, hist / hist.sum(), where='mid', label=label, color=color, linestyle='--', zorder=100)
     else:
         plt.bar(theta_vals, hist / hist.sum(), label=label, color=color, width=5, alpha=0.5)
plt.legend()
plt.title('Orientation')
plt.show()

colors = ['b', 'g', 'r']*2
for label, color in zip(sat_diff_dic, colors):
    hist = sat_diff_dic[label]
    if label[-4:] == '_nwr':
        plt.step(diff_vals,  hist / hist.sum(), where='mid', label=label, color=color, linestyle='--', zorder=100)
    else:
        plt.bar(diff_vals, hist / hist.sum(), label=label, color=color, width=5, alpha=0.5)

plt.legend()
plt.title('Sat. orientation rel. to wind direction at different heights')
plt.show()

