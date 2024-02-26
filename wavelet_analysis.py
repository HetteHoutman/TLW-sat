import datetime as dt
import glob
import numpy as np

import pandas as pd
import py_cwt2d
from fourier import *
from miscellaneous import *
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
    test = True
    stripe_test = False

    pixels_per_km = 1
    lambda_min = 3
    lambda_max = 35
    theta_bin_width = 5
    omega_0x = 6
    pspec_threshold = 5e-4
    # pspec_threshold = 1e-4

    block_size = 50*pixels_per_km + 1
    n_lambda = 60

    # settings
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

    # enhance image
    orig -= orig.min()
    orig /= orig.max()
    # orig = orig > threshold_local(orig, block_size, method='gaussian')
    from skimage.exposure import equalize_hist
    orig = equalize_hist(orig)

    lambdas, lambdas_edges = k_spaced_lambda([lambda_min, lambda_max], n_lambda)
    thetas = np.arange(0, 180, theta_bin_width)
    thetas_edges = create_bins_from_midpoints(thetas)
    scales = lambdas * (omega_0x + np.sqrt(2 + omega_0x**2))/ (4 * np.pi) * pixels_per_km

    # initialise wavelet power spectrum array and fill
    pspec = np.zeros((*orig.shape, len(lambdas), len(thetas)))
    for i, theta in enumerate(thetas):
        cwt, wavnorm = py_cwt2d.cwt_2d(orig, scales, 'morlet', omega_0x=omega_0x, phi=np.deg2rad(90 + theta), epsilon=1)
        pspec[..., i] = (abs(cwt) / scales) ** 2

    # calculate derived things
    pspec = np.ma.masked_less(pspec, pspec_threshold)

    # e-folding distance for Morlet
    efold_dist = np.sqrt(2) * scales
    coi_mask = cone_of_influence_mask(pspec.data, efold_dist, pixels_per_km)
    pspec = np.ma.masked_where(pspec.mask | coi_mask, pspec.data)

    threshold_mask_idx = np.argwhere(~pspec.mask)
    strong_lambdas, strong_thetas = lambdas[threshold_mask_idx[:, -2]], thetas[threshold_mask_idx[:, -1]]

    max_pspec = pspec.max((-2, -1))
    max_lambdas, max_thetas = max_lambda_theta(pspec, lambdas, thetas)

    avg_pspec = np.ma.masked_less(pspec.data, pspec_threshold / 2).mean((0, 1))

    # histograms
    strong_hist, _, _ = np.histogram2d(strong_lambdas, strong_thetas, bins=[lambdas_edges, thetas_edges])
    # strong_hist /= np.repeat(lambdas[..., np.newaxis],  len(thetas), axis=1)
    max_hist, _, _ = np.histogram2d(max_lambdas.flatten(), max_thetas.flatten(), bins=[lambdas_edges, thetas_edges])

    # histogram smoothing (tile along theta-axis and select middle part so that smoothing is periodic over theta
    strong_hist_smoothed = gaussian(np.tile(strong_hist, 3))[:, strong_hist.shape[1]:strong_hist.shape[1] * 2]
    max_hist_smoothed = gaussian(np.tile(max_hist, 3))[:, max_hist.shape[1]:max_hist.shape[1] * 2]

    # determine maximum in smoothed histogram
    lambda_selected, theta_selected, lambda_bounds, theta_bounds = find_polar_max_and_error(strong_hist_smoothed,
                                                                                            lambdas, thetas)

    # plot images
    # TODO for some reason a negative value/level is sometimes passed to colorbar here
    plot_contour_over_image(orig, max_pspec, Lx, Ly, cbarlabel='Maximum of wavelet power spectrum',
                            alpha=0.5, norm='log')
    plt.savefig(save_path + 'wavelet_pspec_max.png', dpi=300)
    plt.close()

    plot_contour_over_image(orig, max_lambdas, Lx, Ly, cbarlabel='Dominant wavelength (km)',
                            alpha=0.5)
    plt.savefig(save_path + 'wavelet_dom_lambda.png', dpi=300)
    plt.close()

    plot_contour_over_image(orig, max_thetas, Lx, Ly, cbarlabel='Dominant orientation (degrees from North)',
                            alpha=0.5)
    plt.savefig(save_path + 'wavelet_dom_theta.png', dpi=300)
    plt.close()

    # plot histograms
    plot_k_histogram(max_lambdas.flatten(), max_thetas.flatten(), lambdas_edges, thetas_edges)
    plt.savefig(save_path + 'wavelet_k_histogram_max.png', dpi=300)
    plt.close()

    plot_k_histogram(strong_lambdas, strong_thetas, lambdas_edges, thetas_edges)
    plt.scatter(lambda_selected, theta_selected, marker='x', color='k')
    plt.savefig(save_path + 'wavelet_k_histogram_full_pspec.png', dpi=300)
    plt.close()

    plot_polar_pcolormesh(strong_hist, lambdas_edges, thetas_edges, cbarlabel='Dominant wavelet count')
    plt.scatter(np.deg2rad(theta_selected), lambda_selected, marker='x', color='k')
    plt.savefig(save_path + 'wavelet_k_histogram_strong_pspec_polar.png', dpi=300)
    plt.close()

    plot_polar_pcolormesh(max_hist, lambdas_edges, thetas_edges, cbarlabel='Dominant wavelet count')
    plt.savefig(save_path + 'wavelet_k_histogram_max_pspec_polar.png', dpi=300)
    plt.close()

    plot_polar_pcolormesh(avg_pspec, lambdas_edges, thetas_edges, cbarlabel='Average power spectrum')
    plt.savefig(save_path + 'wavelet_average_pspec.png', dpi=300)
    plt.close()

    # save results
    if not test:
        csv_root = '../tephiplot/wavelet_results/'
        csv_file = f'sat_adapt_thresh_{block_size}.csv'

        try:
            df = pd.read_csv(csv_root + csv_file, index_col=[0, 1, 2], parse_dates=[0])
        except FileNotFoundError:
            df = pd.read_csv(csv_root + 'template.csv', index_col=[0, 1, 2], parse_dates=[0])

        df.sort_index(inplace=True)

        df.loc[(str(datetime.date()), region, datetime.hour), 'lambda'] = lambda_selected
        df.loc[(str(datetime.date()), region, datetime.hour), 'lambda_min'] = lambda_bounds[0]
        df.loc[(str(datetime.date()), region, datetime.hour), 'lambda_max'] = lambda_bounds[1]
        df.loc[(str(datetime.date()), region, datetime.hour), 'theta'] = theta_selected
        df.loc[(str(datetime.date()), region, datetime.hour), 'theta_min'] = theta_bounds[0]
        df.loc[(str(datetime.date()), region, datetime.hour), 'theta_max'] = theta_bounds[1]

        df.sort_index(inplace=True)
        df.to_csv(csv_root + csv_file)

        # data_root = f'/storage/silver/metstudent/phd/sw825517/sat_pspecs/{datetime.strftime("%Y-%m-%d_%H")}/{region}/'
        # if not os.path.exists(data_root):
        #     os.makedirs(data_root)
        # np.save(data_root + 'pspec.npy', pspec.data)
        # np.save(data_root + 'lambdas.npy', lambdas)
        # np.save(data_root + 'thetas.npy', thetas)
        # np.save(data_root + 'threshold.npy', [pspec_threshold])
        # np.save(data_root + 'histogram.npy', strong_hist)
