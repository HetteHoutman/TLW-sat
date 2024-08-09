import iris
import matplotlib.pyplot as plt
import sys

import datetime as dt
import glob
import numpy as np

import pandas as pd
import py_cwt2d
from matplotlib.patches import Rectangle
from satpy import Scene
from skimage.feature import peak_local_max

from fourier import *
from miscellaneous import *
from prepare_data import get_w_field_img
from prepare_metadata import get_sat_map_bltr
from skimage.filters import gaussian, threshold_local
from wavelet import *
from wavelet_plot import *

from file_to_image import produce_scene
from satpy.writers import get_enhanced_image


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

def add_polar_hist(fig, ax, hist):
    T, L = np.meshgrid(np.deg2rad(thetas_edges), lambdas_edges)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_thetalim((np.deg2rad(thetas_edges[0]), np.deg2rad(thetas_edges[-1])))
    ax.set_rscale('log')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_rlim([int(lambdas_edges[0]), int(lambdas_edges[-1]) + 1])
    ax.set_rgrids([int(lambdas_edges[0]) + 1, 10, 20, 30])

    pc = ax.pcolormesh(T, L, np.ma.masked_equal(hist, 0) / np.repeat(lambdas[..., np.newaxis],  len(thetas), axis=1) **2)
    fig.colorbar(pc, label=r'Area / $\lambda^2$', pad=-0.075, fraction=0.05)
    ax.set_ylabel('Wavelength (km)', labelpad=-20)


if __name__ == '__main__':
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

    datetime_string = '2023-12-17_13'
    datetime = dt.datetime.strptime(datetime_string, '%Y-%m-%d_%H')
    region = 'cornwall'

    seviri, Lx, Ly = get_seviri_img(datetime, region, stripe_test=stripe_test, pixels_per_km=pixels_per_km)
    w, u, v, wind_dir, _, _ = get_w_field_img(datetime, region, leadtime=leadtime, coord=vertical_coord,
                                              map_height=analysis_level)
    ukv = w[0, ::-1].data

    data_root = "/storage/silver/metstudent/phd/sw825517/seviri_data/"
    sat_file_root = data_root + datetime.strftime("%Y-%m-%d_%H") + '/'
    sat_file = find_sat_file(sat_file_root)

    sat_bl, sat_tr, _, _ = get_sat_map_bltr(region, region_root='../tephiplot/regions/')

    scene, a = produce_scene(sat_file_root + sat_file, bottomleft=sat_bl, topright=sat_tr, grid='km',
                             pixels_per_km=pixels_per_km)

    modis_scene = Scene(filenames=glob.glob('../*.hdf'), reader_kwargs={'fill_disk': True}, reader='modis_l1b')
    scene.load(['HRV'])
    modis_scene.load(['1'])
    modis_scene = modis_scene.resample(a)
    modis = np.array(modis_scene['1'].data/100)

    factor = (lambda_max / lambda_min) ** (1 / (n_lambda -1))
    # have two spots before and after lambda range for finding local maxima
    lambdas, lambdas_edges = log_spaced_lambda([lambda_min / factor ** 2, lambda_max * factor ** 2], factor)
    thetas = np.arange(0, 180, theta_bin_width)
    thetas_edges = create_bins_from_midpoints(thetas)
    scales = lambdas * (omega_0x + np.sqrt(2 + omega_0x**2))/ (4 * np.pi) * pixels_per_km

    pspec_ukv = np.zeros((*seviri.shape, len(lambdas), len(thetas)))
    pspec_modis = np.zeros((*seviri.shape, len(lambdas), len(thetas)))
    pspec_seviri = np.zeros((*seviri.shape, len(lambdas), len(thetas)))

    # SEVIRI
    for i, theta in enumerate(thetas):
        cwt, wavnorm = py_cwt2d.cwt_2d(seviri, scales, 'morlet', omega_0x=omega_0x, phi=np.deg2rad(90 + theta), epsilon=1)
        pspec_seviri[..., i] = (abs(cwt) / scales) ** 2

    var = seviri.var()
    pspec_seviri /= var

    threshold_mask = pspec_seviri < pspec_threshold
    efold_dist = np.sqrt(2) * scales
    coi_mask = cone_of_influence_mask(pspec_seviri.data, efold_dist, pixels_per_km)
    pspec_seviri = np.ma.masked_where(threshold_mask | coi_mask, pspec_seviri)
    max_pspec_seviri = pspec_seviri.max((-2, -1))
    max_lambdas_nwr_seviri, max_thetas_nwr_seviri = max_lambda_theta(pspec_seviri, lambdas, thetas)
    max_hist_seviri, _, _ = np.histogram2d(max_lambdas_nwr_seviri[~max_lambdas_nwr_seviri.mask].flatten(), max_thetas_nwr_seviri[~max_lambdas_nwr_seviri.mask].flatten(),
                                           bins=[lambdas_edges, thetas_edges])
    max_hist_smoothed = gaussian(np.tile(max_hist_seviri, 3))[:, max_hist_seviri.shape[1]:max_hist_seviri.shape[1] * 2]
    peak_idxs = peak_local_max(np.tile(max_hist_smoothed, 3), exclude_border=False)
    peak_idxs = peak_idxs[(peak_idxs[:,1] >= max_hist_smoothed.shape[1]) & (peak_idxs[:,1] < max_hist_smoothed.shape[1]*2)]
    peak_idxs[:,1] -= max_hist_smoothed.shape[1]

    # only keep peaks which correspond to an area of larger than area_threshold times lambda^2
    area_threshold = 1
    area_condition = (max_hist_smoothed / np.repeat(lambdas[..., np.newaxis],  len(thetas), axis=1) **2 )[tuple(peak_idxs.T)] > area_threshold
    # area_condition = (max_hist / np.repeat(lambdas[..., np.newaxis],  len(thetas), axis=1) **2 )[tuple(peak_idxs.T)] > area_threshold
    # only keep peaks within lambda range
    lambda_condition = (lambdas[peak_idxs[:,0]] >= 3) & (lambdas[peak_idxs[:,0]] <= 35)

    peak_idxs_seviri = peak_idxs[area_condition & lambda_condition]
    lambdas_selected_seviri, thetas_selected_seviri = lambdas[peak_idxs_seviri[:,0]], thetas[peak_idxs_seviri[:,1]]
    areas_selected_seviri = max_hist_smoothed[tuple(peak_idxs_seviri.T)]

    # MODIS
    for i, theta in enumerate(thetas):
        cwt, wavnorm = py_cwt2d.cwt_2d(modis, scales, 'morlet', omega_0x=omega_0x, phi=np.deg2rad(90 + theta), epsilon=1)
        pspec_modis[..., i] = (abs(cwt) / scales) ** 2

    var = modis.var()
    pspec_modis /= var

    threshold_mask = pspec_modis < pspec_threshold
    efold_dist = np.sqrt(2) * scales
    coi_mask = cone_of_influence_mask(pspec_modis.data, efold_dist, pixels_per_km)

    pspec_modis = np.ma.masked_where(threshold_mask | coi_mask, pspec_modis)
    max_pspec_modis = pspec_modis.max((-2, -1))
    max_lambdas_nwr_modis, max_thetas_nwr_modis = max_lambda_theta(pspec_modis, lambdas, thetas)

    pspec_modis = np.ma.masked_where(threshold_mask | coi_mask, pspec_modis)
    max_pspec_modis = pspec_modis.max((-2, -1))
    max_lambdas_nwr_modis, max_thetas_nwr_modis = max_lambda_theta(pspec_modis, lambdas, thetas)
    max_hist_modis, _, _ = np.histogram2d(max_lambdas_nwr_modis[~max_lambdas_nwr_modis.mask].flatten(),
                                          max_thetas_nwr_modis[~max_lambdas_nwr_modis.mask].flatten(),
                                          bins=[lambdas_edges, thetas_edges])
    max_hist_smoothed = gaussian(np.tile(max_hist_modis, 3))[:, max_hist_modis.shape[1]:max_hist_modis.shape[1] * 2]
    peak_idxs = peak_local_max(np.tile(max_hist_smoothed, 3), exclude_border=False)
    peak_idxs = peak_idxs[
        (peak_idxs[:, 1] >= max_hist_smoothed.shape[1]) & (peak_idxs[:, 1] < max_hist_smoothed.shape[1] * 2)]
    peak_idxs[:, 1] -= max_hist_smoothed.shape[1]

    # only keep peaks which correspond to an area of larger than area_threshold times lambda^2
    area_threshold = 1
    area_condition = (max_hist_smoothed / np.repeat(lambdas[..., np.newaxis], len(thetas), axis=1) ** 2)[
                         tuple(peak_idxs.T)] > area_threshold
    # area_condition = (max_hist / np.repeat(lambdas[..., np.newaxis],  len(thetas), axis=1) **2 )[tuple(peak_idxs.T)] > area_threshold
    # only keep peaks within lambda range
    lambda_condition = (lambdas[peak_idxs[:, 0]] >= 3) & (lambdas[peak_idxs[:, 0]] <= 35)

    peak_idxs_modis = peak_idxs[area_condition & lambda_condition]
    lambdas_selected_modis, thetas_selected_modis = lambdas[peak_idxs_modis[:, 0]], thetas[peak_idxs_modis[:, 1]]
    areas_selected_modis = max_hist_smoothed[tuple(peak_idxs_modis.T)]

    # UKV
    for i, theta in enumerate(thetas):
        cwt, wavnorm = py_cwt2d.cwt_2d(ukv, scales, 'morlet', omega_0x=omega_0x, phi=np.deg2rad(90 + theta), epsilon=1)
        pspec_ukv[..., i] = (abs(cwt) / scales) ** 2

    var = ukv.var()
    pspec_ukv /= var
    threshold_mask = pspec_ukv < pspec_threshold
    efold_dist = np.sqrt(2) * scales
    coi_mask = cone_of_influence_mask(pspec_ukv.data, efold_dist, pixels_per_km)

    pspec_ukv = np.ma.masked_where(threshold_mask | coi_mask, pspec_ukv)
    max_pspec_ukv = pspec_ukv.max((-2, -1))
    max_lambdas_nwr_ukv, max_thetas_nwr_ukv = max_lambda_theta(pspec_ukv, lambdas, thetas)

    pspec_ukv = np.ma.masked_where(threshold_mask | coi_mask, pspec_ukv)
    max_pspec_ukv = pspec_ukv.max((-2, -1))
    max_lambdas_nwr_ukv, max_thetas_nwr_ukv = max_lambda_theta(pspec_ukv, lambdas, thetas)
    max_hist_ukv, _, _ = np.histogram2d(max_lambdas_nwr_ukv[~max_lambdas_nwr_ukv.mask].flatten(),
                                        max_thetas_nwr_ukv[~max_lambdas_nwr_ukv.mask].flatten(),
                                        bins=[lambdas_edges, thetas_edges])
    max_hist_smoothed = gaussian(np.tile(max_hist_ukv, 3))[:, max_hist_ukv.shape[1]:max_hist_ukv.shape[1] * 2]
    peak_idxs = peak_local_max(np.tile(max_hist_smoothed, 3), exclude_border=False)
    peak_idxs = peak_idxs[
        (peak_idxs[:, 1] >= max_hist_smoothed.shape[1]) & (peak_idxs[:, 1] < max_hist_smoothed.shape[1] * 2)]
    peak_idxs[:, 1] -= max_hist_smoothed.shape[1]

    # only keep peaks which correspond to an area of larger than area_threshold times lambda^2
    area_threshold = 1
    area_condition = (max_hist_smoothed / np.repeat(lambdas[..., np.newaxis], len(thetas), axis=1) ** 2)[
                         tuple(peak_idxs.T)] > area_threshold
    # area_condition = (max_hist / np.repeat(lambdas[..., np.newaxis],  len(thetas), axis=1) **2 )[tuple(peak_idxs.T)] > area_threshold
    # only keep peaks within lambda range
    lambda_condition = (lambdas[peak_idxs[:, 0]] >= 3) & (lambdas[peak_idxs[:, 0]] <= 35)

    peak_idxs_ukv = peak_idxs[area_condition & lambda_condition]
    lambdas_selected_ukv, thetas_selected_ukv = lambdas[peak_idxs_ukv[:, 0]], thetas[peak_idxs_ukv[:, 1]]
    areas_selected_ukv = max_hist_smoothed[tuple(peak_idxs_ukv.T)]


    # PLOTTING
    fig = plt.figure(figsize=(6, 15))
    ax00 = fig.add_subplot(3, 2, 1)
    ax00.imshow(modis, cmap='gray')
    ax00.axis('off')
    rect = Rectangle((133, 0), 120, 56, edgecolor='magenta', facecolor='none', linewidth=2)
    ax00.add_patch(rect)

    ax01 = fig.add_subplot(3, 2, 2, projection='polar')
    add_polar_hist(fig, ax01, max_hist_modis)
    for l, t in zip(lambdas_selected_modis, thetas_selected_modis):
        plt.scatter(np.deg2rad(t), l, marker='x', color='r')

    ax10 = fig.add_subplot(3, 2, 3)
    ax10.imshow(seviri, cmap='gray')
    ax10.axis('off')
    rect = Rectangle((133, 0), 120, 56, edgecolor='magenta', facecolor='none', linewidth=2)
    ax10.add_patch(rect)

    ax11 = fig.add_subplot(3, 2, 4, projection='polar')
    add_polar_hist(fig, ax11, max_hist_seviri)
    for l, t in zip(lambdas_selected_seviri, thetas_selected_seviri):
        plt.scatter(np.deg2rad(t), l, marker='x', color='r')

    ax20 = fig.add_subplot(3, 2, 5)
    ax20.imshow(ukv, cmap='gray')
    ax20.axis('off')
    rect = Rectangle((133, 0), 120, 56, edgecolor='magenta', facecolor='none', linewidth=2)
    ax20.add_patch(rect)

    ax21 = fig.add_subplot(3, 2, 6, projection='polar')
    add_polar_hist(fig, ax21, max_hist_ukv)
    for l, t in zip(lambdas_selected_ukv, thetas_selected_ukv):
        plt.scatter(np.deg2rad(t), l, marker='x', color='r')

    ax00.annotate('a)', (0.05, 1.05), xycoords='axes fraction', weight='bold', fontsize=15)
    ax01.annotate('b)', (0.05, 1.05), xycoords='axes fraction', weight='bold', fontsize=15)
    ax10.annotate('c)', (0.05, 1.05), xycoords='axes fraction', weight='bold', fontsize=15)
    ax11.annotate('d)', (0.05, 1.05), xycoords='axes fraction', weight='bold', fontsize=15)
    ax20.annotate('e)', (0.05, 1.05), xycoords='axes fraction', weight='bold', fontsize=15)
    ax21.annotate('f)', (0.05, 1.05), xycoords='axes fraction', weight='bold', fontsize=15)

    plt.tight_layout()
    fig.subplots_adjust(hspace=-0.65)
    plt.savefig('plots/modis_example.png', dpi=300)
    plt.show()


