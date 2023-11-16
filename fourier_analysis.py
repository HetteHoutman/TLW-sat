import sys

import matplotlib.pyplot as plt
import pandas as pd
from astropy.convolution import convolve, Gaussian2DKernel
from fourier import *
from fourier_plot import plot_pspec_polar, plot_radial_pspec, plot_2D_pspec, filtered_inv_plot, plot_corr
from miscellaneous import check_argv_num, load_settings, get_region_var, get_datetime_from_settings, \
    make_title_and_save_path
from psd import periodic_smooth_decomp

from file_to_image import produce_scene


def get_seviri_img(settings, magnitude_filter=False, stripe_test=False):
    scene, crs = produce_scene(settings.sat_file, bottomleft=sat_bl, topright=sat_tr, grid='km')
    Lx, Ly = extract_distances(scene['HRV'].y[::-1], scene['HRV'].x)
    # divide by 100 to convert from % to 0-1
    orig = np.array(scene['HRV'].data) / 100
    # orig -= orig.mean()

    if magnitude_filter:
        orig[(orig > 12) & (orig < 25)] = 6

    if stripe_test:
        orig = stripey_test(orig, Lx, Ly, [10], [15], wiggle=20, wiggle_wavelength=20)

    # perform decomposition to remove cross-like signal
    orig, smooth = periodic_smooth_decomp(orig)

    return orig, Lx, Ly


if __name__ == '__main__':
    # options
    k2 = True
    smoothed = True
    mag_filter = False
    test = False
    stripe_test = False

    min_lambda = 4
    max_lambda = 30
    wnum_bin_width = 0.025
    theta_bin_width = 2.5

    # settings
    check_argv_num(sys.argv, 2, "(settings, region json files)")
    s = load_settings(sys.argv[1])
    datetime = get_datetime_from_settings(s)
    region = sys.argv[2]
    sat_bounds = get_region_var("sat_bounds", region,
                                r"C:/Users/sw825517/OneDrive - University of Reading/research/code/tephi_plot/regions/")
    sat_bl, sat_tr = sat_bounds[:2], sat_bounds[2:]
    my_title, save_path = make_title_and_save_path(datetime, region, 'sat', test, smoothed, mag_filter, k2=k2)

    # produce image
    orig, Lx, Ly = get_seviri_img(s, magnitude_filter=mag_filter, stripe_test=stripe_test)

    # define reciprocal space
    K, L, wavenumbers, thetas = recip_space(Lx, Ly, orig.shape)
    wavelengths = 2 * np.pi / wavenumbers

    # plot histogram of image values
    plt.hist(orig.flatten(), bins=100)
    plt.savefig(save_path + 'hist.png', dpi=300)

    # do actual fourier transform
    ft = np.fft.fft2(orig)
    shifted_ft = np.fft.fftshift(ft)

    # bandpass through expected TLW wavelengths
    bandpassed = ideal_bandpass(shifted_ft, Lx, Ly, 2 * np.pi / max_lambda, 2 * np.pi / min_lambda)

    # plot ingoing data
    plt.figure()
    filtered_inv_plot(orig, bandpassed, Lx, Ly, inverse_fft=False, title=my_title, # latlon=area_extent
                      )
    plt.savefig(save_path + 'sat_plot.png', dpi=300)

    # get 2d power spectrum
    pspec_2d = np.ma.masked_where(bandpassed.mask, abs(shifted_ft) ** 2)

    # multiply by |k|^2
    if k2:
        pspec_2d = np.ma.masked_where(pspec_2d.mask, pspec_2d.data * wavenumbers ** 2)

    # convert power spectrum to polar coordinates
    radial_pspec, wnum_bins, wnum_vals, theta_bins, theta_vals = make_polar_pspec(pspec_2d, wavenumbers, wnum_bin_width,
                                                                                  thetas, theta_bin_width)

    if smoothed:
        pspec_2d = np.ma.masked_where(pspec_2d.mask, convolve(pspec_2d.data, Gaussian2DKernel(7, x_size=15, y_size=15),
                                                              boundary='wrap'))
        radial_pspec = convolve(radial_pspec, Gaussian2DKernel(3, x_size=11, y_size=11), boundary='wrap')

    # find maximum in polar power spectrum
    bounded_polar_pspec, bounded_wnum_vals = apply_wnum_bounds(radial_pspec, wnum_vals, wnum_bins,
                                                               (min_lambda, max_lambda))

    dominant_wnum, dominant_theta = find_max(bounded_polar_pspec, bounded_wnum_vals, theta_vals)

    # plot polar power spectrum along with maximum
    plt.figure()
    plot_pspec_polar(wnum_bins, theta_bins, radial_pspec, scale='log', xlim=(0.05, 4.5),
                     vmin=np.nanmin(bounded_polar_pspec), vmax=np.nanmax(bounded_polar_pspec),
                     title=my_title, min_lambda=min_lambda, max_lambda=max_lambda)
    plt.scatter(dominant_wnum, dominant_theta, marker='x', color='k', s=100, zorder=100)
    plt.tight_layout()
    plt.savefig(save_path + 'polar_pspec.png', dpi=300)

    print(f'Dominant wavelength by maximum of power spectrum: {2 * np.pi / dominant_wnum:.2f} km')
    print(f'Dominant angle by maximum of power spectrum: {dominant_theta:.0f} deg from north')

    # plot radial power spectrum
    plt.figure()
    plot_radial_pspec(radial_pspec, wnum_vals, theta_bins, dominant_wnum, title=my_title)
    plt.savefig(save_path + 'radial_pspec.png', dpi=300)

    # perform correlation with ellipse
    collapsed_corr = get_ellipse_correlation(pspec_2d, thetas, (2,25))

    # find maximum in correlation array
    dominant_wlen, dominant_theta, dom_K, dom_L = find_corr_max(collapsed_corr, K, L, wavelengths, thetas)
    dominant_wnum = 2*np.pi/dominant_wlen
    (lambda_min, lambda_plus), (theta_min, theta_plus) = find_corr_error(collapsed_corr, K, L, 2 * np.pi / dominant_wlen, dominant_theta)

    err_wnums = np.linspace(2*np.pi/lambda_min, 2*np.pi/lambda_plus, 500)
    err_thetas = np.linspace(theta_min, theta_plus, 500)

    # plot correlation in array with maximum
    plt.figure()
    plot_corr(collapsed_corr, K, L)
    # TODO turn into plot errors function
    for i in range(len(err_thetas)):
        plt.scatter(*pol2cart(dominant_wnum, np.deg2rad(-(err_thetas[i] % 180 - 90))), c='k', s=0.5)
        plt.scatter(*pol2cart(err_wnums[i], np.deg2rad(-dominant_theta+90)), c='k', s=0.5)
    plt.scatter(dom_K, dom_L, marker='x')
    plt.savefig(save_path + 'corr.png', dpi=300)

    # plot cartesian power spectrum with maximum from ellipse correlation
    plt.figure()
    plot_2D_pspec(pspec_2d, K, L, wavelengths, wavelength_contours=[5, 10, 35], title=my_title)
    plt.scatter(dom_K, dom_L, marker='x')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.savefig(save_path + '2d_pspec_withcross.png', dpi=300)
    plt.close()

    print(f'Dominant wavelength by ellipse method: {dominant_wlen:.2f} km')
    print(f'Dominant wavelength range estimate: {lambda_min:.2f}-{lambda_plus:.2f} km')
    print(f'Dominant angle by ellipse method: {dominant_theta:.0f} deg from north')
    print(f'Dominant angle range estimate: {theta_min:.0f}-{theta_plus:.0f} deg')

    # save results to csv
    if not test:
        csv_root = '../../other_data/fourier_results/'
        csv_file = 'sat'
        if k2:
            csv_file += '_k2'
        if smoothed:
            csv_file += '_smoothed'

        csv_file += '_results.csv'

        try:
            df = pd.read_csv(csv_root + csv_file, index_col=[0, 1, 2])
        except FileNotFoundError:
            df = pd.read_csv(csv_root + 'template.csv', index_col=[0, 1, 2])

        df.sort_index(inplace=True)
        date = f'{s.year}-{s.month:02d}-{s.day:02d}'

        df.loc[(date, region, s.h), 'lambda'] = dominant_wlen
        df.loc[(date, region, s.h), 'lambda_min'] = lambda_min
        df.loc[(date, region, s.h), 'lambda_max'] = lambda_plus
        df.loc[(date, region, s.h), 'theta'] = dominant_theta
        df.loc[(date, region, s.h), 'theta_min'] = theta_min
        df.loc[(date, region, s.h), 'theta_max'] = theta_plus

        # sort for clarity if any new dates have been added
        df.sort_index(inplace=True)
        df.to_csv(csv_root + csv_file)

    np.save(save_path + 'pspec_array.npy', pspec_2d.data)

