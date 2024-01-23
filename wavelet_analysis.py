import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py_cwt2d
from fourier import *
from miscellaneous import *
from psd import periodic_smooth_decomp
from wavelet_plot import *
from wavelet import *
from skimage import exposure
from skimage.filters import gaussian

from file_to_image import produce_scene


def get_seviri_img(settings, sat_bl, sat_tr, stripe_test=False):
    scene, crs = produce_scene(settings.sat_file, bottomleft=sat_bl, topright=sat_tr, grid='km')
    Lx, Ly = extract_distances(scene['HRV'].y[::-1], scene['HRV'].x)

    # divide by 100 to convert from % to 0-1
    orig = np.array(scene['HRV'].data) / 100

    if stripe_test:
        orig = stripey_test(orig, Lx, Ly, [10, 30], [15, 100], wiggle=0, wiggle_wavelength=20)

    # perform decomposition to remove cross-like signal
    orig, smooth = periodic_smooth_decomp(orig)

    return orig, Lx, Ly


def run(args):
    # options
    test = False
    stripe_test = False

    lambda_min = 4
    lambda_max = 30
    lambda_bin_width = 1
    theta_bin_width = 5
    omega_0x = 6
    pspec_threshold = 3e-4
    # pspec_threshold = 2e-3

    # settings
    check_argv_num(args, 2, "(settings, region json files)")
    s = load_settings(args[1])
    datetime = get_datetime_from_settings(s)
    region = args[2]
    sat_bounds = get_region_var("sat_bounds", region,
                                r"C:/Users/sw825517/OneDrive - University of Reading/research/code/tephi_plot/regions/")
    sat_bl, sat_tr = sat_bounds[:2], sat_bounds[2:]
    save_path = f'./plots/{datetime}/{region}/'
    if test:
        save_path = f'./plots/test/'

    # produce image
    orig, Lx, Ly = get_seviri_img(s, sat_bl, sat_tr, stripe_test=stripe_test)

    # normalise orig image
    orig /= orig.max()
    # orig = exposure.equalize_adapthist(orig, clip_limit=0.03)

    # define evaluation space
    lambdas = np.arange(lambda_min, lambda_max + 1)
    lambdas_edges = create_bins_from_midpoints(lambdas)
    thetas = np.arange(0, 180, theta_bin_width)
    thetas_edges = create_bins_from_midpoints(thetas)
    scales = omega_0x * lambdas / (2 * np.pi)

    # initialise wavelet power spectrum array and fill
    pspec = np.zeros((*orig.shape, len(lambdas), len(thetas)))
    for i, theta in enumerate(thetas):
        cwt, wavnorm = py_cwt2d.cwt_2d(orig, scales, 'morlet', omega_0x=omega_0x, phi=np.deg2rad(90 + theta))
        pspec[..., i] = (abs(cwt) / scales / wavnorm) ** 2

    # calculate derived things
    pspec = np.ma.masked_less(pspec, pspec_threshold)
    threshold_mask_idx = np.argwhere(~pspec.mask)
    strong_lambdas, strong_thetas = lambdas[threshold_mask_idx[:, -2]], thetas[threshold_mask_idx[:, -1]]

    max_pspec = np.ma.masked_less(pspec.data.max((-2, -1)), pspec_threshold)
    max_lambdas, max_thetas = max_lambda_theta(pspec.data, lambdas, thetas)

    avg_pspec = pspec.data.mean((0, 1))

    # histograms
    strong_hist, _, _ = np.histogram2d(strong_lambdas, strong_thetas, bins=[lambdas_edges, thetas_edges])
    max_hist, _, _ = np.histogram2d(max_lambdas[~max_pspec.mask], max_thetas[~max_pspec.mask],
                                    bins=[lambdas_edges, thetas_edges])

    # histogram smoothing (tile along theta-axis and select middle part so that smoothing is periodic over theta
    strong_hist_smoothed = gaussian(np.tile(strong_hist, 3))[:, strong_hist.shape[1]:strong_hist.shape[1] * 2]
    max_hist_smoothed = gaussian(np.tile(max_hist, 3))[:, max_hist.shape[1]:max_hist.shape[1] * 2]

    # determine maximum in smoothed image
    idxs = np.unravel_index(strong_hist_smoothed.argmax(), strong_hist_smoothed.shape)
    result_lambda, result_theta = lambdas[idxs[0]], thetas[idxs[1]]

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
    plt.scatter(result_lambda, result_theta, marker='x', color='k')
    plt.savefig(save_path + 'wavelet_k_histogram_full_pspec.png', dpi=300)
    plt.show()

    plot_polar_pcolormesh(strong_hist, lambdas_edges, thetas_edges, cbarlabel='Dominant wavelet count')
    plt.scatter(np.deg2rad(result_theta), result_lambda, marker='x', color='k')
    plt.savefig(save_path + 'wavelet_k_histogram_strong_pspec_polar.png', dpi=300)
    plt.show()

    plot_polar_pcolormesh(max_hist, lambdas_edges, thetas_edges, cbarlabel='Dominant wavelet count')
    plt.savefig(save_path + 'wavelet_k_histogram_max_pspec_polar.png', dpi=300)
    plt.show()

    # save results
    if not test:
        csv_root = '../../other_data/wavelet_results/'
        csv_file = 'sat_hist.csv'

        try:
            df = pd.read_csv(csv_root + csv_file, index_col=[0, 1, 2], parse_dates=[0], dayfirst=True)
        except FileNotFoundError:
            df = pd.read_csv(csv_root + 'template.csv', index_col=[0, 1, 2], parse_dates=[0], dayfirst=True)

        df.sort_index(inplace=True)
        date = pd.to_datetime(f'{s.year}-{s.month:02d}-{s.day:02d}')

        df.loc[(date, region, s.h), 'lambda'] = result_lambda
        df.loc[(date, region, s.h), 'theta'] = result_theta

        df.sort_index(inplace=True)
        df.to_csv(csv_root + csv_file)


if __name__ == '__main__':
    run(sys.argv)
