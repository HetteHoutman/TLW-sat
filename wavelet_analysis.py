import datetime as dt
import glob

import pandas as pd
import py_cwt2d
from prepare_metadata import get_sat_map_bltr
from skimage.filters import gaussian, threshold_local

from file_to_image import produce_scene
from fourier import *
from miscellaneous import *
from psd import periodic_smooth_decomp
from wavelet import *
from wavelet_plot import *


def find_sat_file(root):
    return glob.glob('*.nat', root_dir=root)[0]


def get_seviri_img(datetime, region, stripe_test=False,
                   data_root="C:\\Users\\sw825517\\OneDrive - University of Reading\\research\\code\\eumetsat\\data\\"):

    sat_file_root = data_root + datetime.strftime("%Y-%m-%d_%H") + '/'
    sat_file = find_sat_file(sat_file_root)

    sat_bl, sat_tr, _, _ = get_sat_map_bltr(region, region_root='../tephi_plot/regions/')

    scene, crs = produce_scene(sat_file_root + sat_file, bottomleft=sat_bl, topright=sat_tr, grid='km')
    Lx, Ly = extract_distances(scene['HRV'].y[::-1], scene['HRV'].x)

    # divide by 100 to convert from % to 0-1
    orig = np.array(scene['HRV'].data) / 100

    if stripe_test:
        orig = stripey_test(orig, Lx, Ly, [10, 30], [15, 100], wiggle=0, wiggle_wavelength=20)

    # perform decomposition to remove cross-like signal
    orig, smooth = periodic_smooth_decomp(orig)

    return orig, Lx, Ly


if __name__ == '__main__':
    # options
    test = False
    stripe_test = False

    lambda_min = 5
    lambda_max = 35
    lambda_bin_width = 1
    theta_bin_width = 5
    omega_0x = 6
    pspec_threshold = 1e-2
    block_size = 51

    # settings
    check_argv_num(sys.argv, 2, "(datetime (YYYY-MM-DD_HH), region)")
    datetime_string = sys.argv[1]
    datetime = dt.datetime.strptime(datetime_string, '%Y-%m-%d_%H')
    region = sys.argv[2]

    save_path = f'./plots/{datetime_string}/{region}/51_'
    if test:
        save_path = f'./plots/test/'

    # produce image
    orig, Lx, Ly = get_seviri_img(datetime, region, stripe_test=stripe_test)

    # enhance image
    orig = orig > threshold_local(orig, block_size, method='gaussian')

    lambdas, lambdas_edges = k_spaced_lambda([lambda_min, lambda_max], 40)
    thetas = np.arange(0, 180, theta_bin_width)
    thetas_edges = create_bins_from_midpoints(thetas)
    scales = omega_0x * lambdas / (2 * np.pi)

    # initialise wavelet power spectrum array and fill
    pspec = np.zeros((*orig.shape, len(lambdas), len(thetas)))
    for i, theta in enumerate(thetas):
        cwt, wavnorm = py_cwt2d.cwt_2d(orig, scales, 'morlet', omega_0x=omega_0x, phi=np.deg2rad(90 + theta), epsilon=1)
        pspec[..., i] = (abs(cwt) / scales / wavnorm) ** 2

    # calculate derived things
    pspec = np.ma.masked_less(pspec, pspec_threshold)
    threshold_mask_idx = np.argwhere(~pspec.mask)
    strong_lambdas, strong_thetas = lambdas[threshold_mask_idx[:, -2]], thetas[threshold_mask_idx[:, -1]]

    max_pspec = np.ma.masked_less(pspec.data.max((-2, -1)), pspec_threshold)
    max_lambdas, max_thetas = max_lambda_theta(pspec.data, lambdas, thetas)

    avg_pspec = np.ma.masked_less(pspec.data, pspec_threshold / 2).mean((0, 1))

    # histograms
    strong_hist, _, _ = np.histogram2d(strong_lambdas, strong_thetas, bins=[lambdas_edges, thetas_edges])
    max_hist, _, _ = np.histogram2d(max_lambdas[~max_pspec.mask], max_thetas[~max_pspec.mask],
                                    bins=[lambdas_edges, thetas_edges])

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
    plt.show()

    plot_contour_over_image(orig, max_lambdas, Lx, Ly, cbarlabel='Dominant wavelength (km)',
                            alpha=0.5)
    plt.savefig(save_path + 'wavelet_dom_lambda.png', dpi=300)
    plt.show()

    plot_contour_over_image(orig, max_thetas, Lx, Ly, cbarlabel='Dominant orientation (degrees from North)',
                            alpha=0.5)
    plt.savefig(save_path + 'wavelet_dom_theta.png', dpi=300)
    plt.show()

    # plot histograms
    plot_k_histogram(max_lambdas[~max_pspec.mask], max_thetas[~max_pspec.mask], lambdas_edges, thetas_edges)
    plt.savefig(save_path + 'wavelet_k_histogram_max.png', dpi=300)
    plt.show()

    plot_k_histogram(strong_lambdas, strong_thetas, lambdas_edges, thetas_edges)
    plt.scatter(lambda_selected, theta_selected, marker='x', color='k')
    plt.savefig(save_path + 'wavelet_k_histogram_full_pspec.png', dpi=300)
    plt.show()

    plot_polar_pcolormesh(strong_hist, lambdas_edges, thetas_edges, cbarlabel='Dominant wavelet count')
    plt.scatter(np.deg2rad(theta_selected), lambda_selected, marker='x', color='k')
    plt.savefig(save_path + 'wavelet_k_histogram_strong_pspec_polar.png', dpi=300)
    plt.show()

    plot_polar_pcolormesh(max_hist, lambdas_edges, thetas_edges, cbarlabel='Dominant wavelet count')
    plt.savefig(save_path + 'wavelet_k_histogram_max_pspec_polar.png', dpi=300)
    plt.show()

    plot_polar_pcolormesh(avg_pspec, lambdas_edges, thetas_edges, cbarlabel='Average power spectrum')
    plt.savefig(save_path + 'wavelet_average_pspec.png', dpi=300)
    plt.show()

    # save results
    if not test:
        csv_root = '../../other_data/wavelet_results/'
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
