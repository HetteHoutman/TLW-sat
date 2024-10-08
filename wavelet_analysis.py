import iris
import matplotlib.pyplot as plt
import sys

import datetime as dt
import glob
import numpy as np

import pandas as pd
import py_cwt2d
from skimage.feature import peak_local_max

from fourier import *
from miscellaneous import *
from prepare_data import get_w_field_img
from prepare_metadata import get_sat_map_bltr
from skimage.filters import gaussian, threshold_local
from wavelet import *
from wavelet_plot import *

from file_to_image import produce_scene


def find_sat_file(root):
    return glob.glob('*.nat', root_dir=root)[0]


def get_seviri_img(datetime, region, stripe_test=False, pixels_per_km=1,
                   data_root="/storage/silver/metstudent/phd/sw825517/seviri_data/"):

    sat_file_root = data_root + datetime.strftime("%Y-%m-%d_%H") + '/'
    sat_file = find_sat_file(sat_file_root)

    sat_bl, sat_tr, _, _ = get_sat_map_bltr(region, region_root='../tephiplot/regions/')

    scene, crs = produce_scene(sat_file_root + sat_file, bottomleft=sat_bl, topright=sat_tr, grid='km',
                               pixels_per_km=pixels_per_km)
    Lx, Ly = extract_distances(scene['HRV'].y[::-1], scene['HRV'].x)

    # divide by 100 to convert from % to 0-1
    orig = np.array(scene['HRV'].data) / 100

    if stripe_test:
        orig = stripey_test(orig, Lx, Ly, [4,4], [30, 90], wiggle=0, wiggle_wavelength=20)

    return orig, Lx, Ly


if __name__ == '__main__':
    # options
    test = False
    stripe_test = False

    pixels_per_km = 1
    lambda_min = 3
    lambda_max = 35
    theta_bin_width = 5
    omega_0x = 6
    wind_deviation_thresh = 50
    pspec_threshold = 1e-2

    leadtime = 0
    block_size = 50*pixels_per_km + 1
    vertical_coord = 'air_pressure'
    analysis_level = 80000
    n_lambda = 50

    # settings
    print('Running ' + sys.argv[1] + ' ' + sys.argv[2])
    check_argv_num(sys.argv, 2, "(datetime (YYYY-MM-DD_HH), region)")
    datetime_string = sys.argv[1]
    datetime = dt.datetime.strptime(datetime_string, '%Y-%m-%d_%H')
    region = sys.argv[2]

    save_path = f'./plots/{datetime_string}/{region}/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if test:
        save_path = f'./plots/test/'

    # produce image
    orig, Lx, Ly = get_seviri_img(datetime, region, stripe_test=stripe_test, pixels_per_km=pixels_per_km)
    w, u, v, wind_dir, _, _ = get_w_field_img(datetime, region, leadtime=leadtime, coord=vertical_coord,
                                              map_height=analysis_level)

    factor = (lambda_max / lambda_min) ** (1 / (n_lambda -1))
    # have two spots before and after lambda range for finding local maxima
    lambdas, lambdas_edges = log_spaced_lambda([lambda_min / factor ** 2, lambda_max * factor ** 2], factor)
    thetas = np.arange(0, 180, theta_bin_width)
    thetas_edges = create_bins_from_midpoints(thetas)
    scales = lambdas * (omega_0x + np.sqrt(2 + omega_0x**2))/ (4 * np.pi) * pixels_per_km

    # initialise wavelet power spectrum array and fill
    pspec = np.zeros((*orig.shape, len(lambdas), len(thetas)))
    for i, theta in enumerate(thetas):
        cwt, wavnorm = py_cwt2d.cwt_2d(orig, scales, 'morlet', omega_0x=omega_0x, phi=np.deg2rad(90 + theta), epsilon=1)
        pspec[..., i] = (abs(cwt) / scales) ** 2

    var = orig.var()
    pspec /= var

    # exclude points: (1) less than threshold; (2) within COI; and (3) more than 50 deg from local UKV wind direction
    threshold_mask = pspec < pspec_threshold
    efold_dist = np.sqrt(2) * scales
    coi_mask = cone_of_influence_mask(pspec.data, efold_dist, pixels_per_km)

    pspec = np.ma.masked_where(threshold_mask | coi_mask, pspec)

    # calculate derived things
    # strong stuff does not include wind restriction
    threshold_mask_idx = np.argwhere(~pspec.mask)
    strong_lambdas, strong_thetas = lambdas[threshold_mask_idx[:, -2]], thetas[threshold_mask_idx[:, -1]]

    max_pspec = pspec.max((-2, -1))
    max_lambdas_nwr, max_thetas_nwr = max_lambda_theta(pspec, lambdas, thetas)

    # now also exclude points that deviate too far from wind direc (but not for max_pspec so that you can keep old mask)
    wind_mask = ((wind_dir.data[::-1] % 180 - max_thetas_nwr.data) % 180 > wind_deviation_thresh) & \
                ((max_thetas_nwr.data - wind_dir.data[::-1] % 180) % 180 > wind_deviation_thresh)

    max_lambdas = np.ma.masked_where(max_lambdas_nwr.mask | wind_mask, max_lambdas_nwr)
    max_thetas = np.ma.masked_where(max_thetas_nwr.mask | wind_mask, max_thetas_nwr)

    # calculate histograms
    strong_hist, _, _ = np.histogram2d(strong_lambdas, strong_thetas, bins=[lambdas_edges, thetas_edges])
    max_hist, _, _ = np.histogram2d(max_lambdas[~max_lambdas.mask].flatten(), max_thetas[~max_lambdas.mask].flatten(), bins=[lambdas_edges, thetas_edges])

    # histogram smoothing (tile along theta-axis and select middle part so that smoothing is periodic over theta
    strong_hist_smoothed = gaussian(np.tile(strong_hist, 3))[:, strong_hist.shape[1]:strong_hist.shape[1] * 2]
    max_hist_smoothed = gaussian(np.tile(max_hist, 3))[:, max_hist.shape[1]:max_hist.shape[1] * 2]

    # find peaks
    peak_idxs = peak_local_max(np.tile(max_hist_smoothed, 3), exclude_border=False)
    peak_idxs = peak_idxs[(peak_idxs[:,1] >= max_hist_smoothed.shape[1]) & (peak_idxs[:,1] < max_hist_smoothed.shape[1]*2)]
    peak_idxs[:,1] -= max_hist_smoothed.shape[1]

    # only keep peaks which correspond to an area of larger than area_threshold times lambda^2
    area_threshold = 1
    area_condition = (max_hist_smoothed / np.repeat(lambdas[..., np.newaxis],  len(thetas), axis=1) **2 )[tuple(peak_idxs.T)] > area_threshold
    # area_condition = (max_hist / np.repeat(lambdas[..., np.newaxis],  len(thetas), axis=1) **2 )[tuple(peak_idxs.T)] > area_threshold
    # only keep peaks within lambda range
    lambda_condition = (lambdas[peak_idxs[:,0]] >= 3) & (lambdas[peak_idxs[:,0]] <= 35)

    peak_idxs = peak_idxs[area_condition & lambda_condition]
    lambdas_selected, thetas_selected = lambdas[peak_idxs[:,0]], thetas[peak_idxs[:,1]]
    areas_selected = max_hist_smoothed[tuple(peak_idxs.T)]
    # areas_selected = max_hist[tuple(peak_idxs.T)]

    # plot images
    plot_wind_and_or(u, v, max_thetas, orig)
    plt.savefig(save_path + 'winds_and_orientations.png', dpi=300)
    plt.close()

    wind_theta_diff = cube_from_array_and_cube((max_thetas[::-1])[None, ...] - wind_dir.data, u)
    wind_theta_nwr_diff = cube_from_array_and_cube((max_thetas_nwr[::-1])[None, ...] - wind_dir.data, u)
    plt.hist(wind_theta_diff[0].data[~wind_theta_diff[0].data.mask], bins=np.arange(-90, 95, 5), label=r'$\vartheta \leq $' + str(wind_deviation_thresh) + r'$\degree$')
    plt.hist(wind_theta_nwr_diff[0].data[~wind_theta_nwr_diff[0].data.mask], bins=np.arange(-90, 95, 5), histtype='step', linestyle='--', color='k', label=r'all $\vartheta$ allowed')
    plt.legend()
    plt.savefig(save_path + 'wind_theta_diff.png', dpi=300)
    plt.close()

    # TODO for some reason a negative value/level is sometimes passed to colorbar here
    plot_contour_over_image(orig, max_pspec, Lx, Ly, cbarlabels=[r'TOA reflectance', r'$\max$ $P(\lambda, \vartheta)/\sigma^2$'],
                            pspec_thresh=pspec_threshold, alpha=0.5)
    plt.savefig(save_path + 'wavelet_pspec_max.png', dpi=300)
    plt.close()

    plot_contour_over_image(orig, max_lambdas, Lx, Ly, cbarlabels=[r'Vertical velocity $\mathregular{(ms^{-1})}$', 'Dominant wavelength (km)'],
                            alpha=0.5)
    plt.savefig(save_path + 'wavelet_dom_lambda.png', dpi=300)
    plt.close()

    plot_contour_over_image(orig, max_thetas, Lx, Ly, cbarlabels=[r'Vertical velocity $\mathregular{(ms^{-1})}$', 'Dominant orientation (degrees from North)'],
                            alpha=0.5)
    plt.savefig(save_path + 'wavelet_dom_theta.png', dpi=300)
    plt.close()

    # plot histograms
    plot_polar_pcolormesh(np.ma.masked_equal(strong_hist, 0) / np.repeat(lambdas[..., np.newaxis],  len(thetas), axis=1) **2, lambdas_edges, thetas_edges, cbarlabel=r'Area / $\lambda^2$', vmin=0)
    for l, t in zip(lambdas_selected, thetas_selected):
        plt.scatter(np.deg2rad(t), l, marker='x', color='k')
    plt.savefig(save_path + 'wavelet_k_histogram_strong_pspec_polar.png', dpi=300)
    plt.close()

    plot_polar_pcolormesh(np.ma.masked_equal(max_hist, 0) / np.repeat(lambdas[..., np.newaxis],  len(thetas), axis=1) **2, lambdas_edges, thetas_edges, cbarlabel=r'Area / $\lambda^2$', vmin=0)
    for l, t in zip(lambdas_selected, thetas_selected):
        plt.scatter(np.deg2rad(t), l, marker='x', color='k')
    plt.savefig(save_path + 'wavelet_k_histogram_max_pspec_polar.png', dpi=300)
    plt.close()

    # save results
    if not test:
        csv_root = '../tephiplot/wavelet_results/'
        csv_file = f'sat_final_800.csv'
        try:
            df = pd.read_csv(csv_root + csv_file, parse_dates=[0])
        except FileNotFoundError:
            df = pd.read_csv(csv_root + 'new_template.csv', parse_dates=[0])

        # store peaks
        for l, t, area in zip(lambdas_selected, thetas_selected, areas_selected):
            df.loc[len(df)] = [datetime.date(), region, datetime.hour, l, t, area]

        df.to_csv(csv_root + csv_file, index=False)

        data_root = f'/storage/silver/metstudent/phd/sw825517/sat_data/{datetime.strftime("%Y-%m-%d_%H")}/{region}/'

        if not os.path.exists(data_root):
                os.makedirs(data_root)

        iris.save(wind_dir, data_root + 'wind_dir_800.nc')
        np.save(data_root + 'max_thetas_800.npy', max_thetas.data)
        np.save(data_root + 'max_lambdas_800.npy', max_lambdas.data)
        np.save(data_root + 'mask_800.npy', max_thetas.mask) 
        np.save(data_root + 'max_thetas_nwr_800.npy', max_thetas_nwr.data)
        np.save(data_root + 'max_lambdas_nwr_800.npy', max_lambdas_nwr.data)
        np.save(data_root + 'mask_nwr_800.npy', max_thetas_nwr.mask)
        np.save(data_root + 'std_800.npy', [np.sqrt(var)])
        # np.save(data_root + 'pspec.npy', pspec.data)
        # np.save(data_root + 'lambdas.npy', lambdas)
        # np.save(data_root + 'thetas.npy', thetas)
        # np.save(data_root + 'threshold.npy', [pspec_threshold])
        # np.save(data_root + 'histogram.npy', strong_hist)
