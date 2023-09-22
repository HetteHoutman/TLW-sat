import os
import sys

import matplotlib.pyplot as plt
from fourier import *
from fourier_plot import plot_pspec_polar, plot_radial_pspec, plot_2D_pspec, filtered_inv_plot
from miscellaneous import check_argv_num, load_settings, get_region_var
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve, Gaussian2DKernel

from file_to_image import produce_scene
from psd import periodic_smooth_decomp

if __name__ == '__main__':
    k2 = True
    smoothed = True

    check_argv_num(sys.argv, 2, "(settings, region json files)")
    s = load_settings(sys.argv[1])
    datetime = f'{s.year}-{s.month}-{s.day}_{s.h}'

    sat_bounds = get_region_var("sat_bounds", sys.argv[2],
                                r"C:/Users/sw825517/OneDrive - University of Reading/research/code/tephi_plot/regions/")
    sat_bl, sat_tr = sat_bounds[:2], sat_bounds[2:]

    if not os.path.exists('plots/' + datetime):
        os.makedirs('plots/' + datetime)

    save_path = f'plots/{datetime}/{sys.argv[2]}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    my_title = f'{datetime}_{sys.argv[2]}_sat'

    save_path = f'plots/test/'
    my_title += '_test'

    if k2:
        save_path += 'k2_'
        my_title += '_k2'

    if smoothed:
        save_path += 'smoothed_'
        my_title += '_smoothed'

    scene, crs = produce_scene(s.sat_file, bottomleft=sat_bl, topright=sat_tr, grid='km')
    Lx, Ly = extract_distances(scene['HRV'].y[::-1], scene['HRV'].x)
    orig = np.array(scene['HRV'].data)
    # orig -= orig.mean()

    K, L, wavenumbers, thetas = recip_space(Lx, Ly, orig.shape)
    wavelengths = 2 * np.pi / wavenumbers

    # orig = stripey_test(orig, Lx, Ly, [10], [15], wiggle=20, wiggle_wavelength=20)
    orig, s = periodic_smooth_decomp(orig)

    ft = np.fft.fft2(orig)
    shifted_ft = np.fft.fftshift(ft)

    min_lambda = 4
    max_lambda = 25
    bandpassed = ideal_bandpass(shifted_ft, Lx, Ly, 2 * np.pi / max_lambda, 2 * np.pi / min_lambda)
    filtered_inv_plot(orig, bandpassed, Lx, Ly, inverse_fft=True, title=my_title
                      # latlon=area_extent
                      )
    plt.savefig(save_path + 'sat_plot.png', dpi=300)
    # plt.show()
    plt.figure()

    # TODO check if this is mathematically the right way of calculating pspec
    pspec_2d = np.ma.masked_where(bandpassed.mask, abs(shifted_ft) ** 2)

    if k2:
        pspec_2d = np.ma.masked_where(pspec_2d.mask, pspec_2d.data * wavenumbers ** 2)

    wnum_bin_width = 0.025
    theta_bin_width = 2.5
    radial_pspec, wnum_bins, wnum_vals, theta_bins, theta_vals = make_polar_pspec(pspec_2d, wavenumbers, wnum_bin_width,
                                                                                  thetas, theta_bin_width)
    if smoothed:
        pspec_2d = np.ma.masked_where(pspec_2d.mask, convolve(pspec_2d.data, Gaussian2DKernel(7, x_size=15, y_size=15), boundary='wrap'))
        radial_pspec = convolve(radial_pspec, Gaussian2DKernel(3, x_size=11, y_size=11), boundary='wrap')

    plot_2D_pspec(pspec_2d, Lx, Ly, wavelength_contours=[5, 10, 35], title=my_title)
    plt.savefig(save_path + '2d_pspec.png', dpi=300)
    # plt.show()
    plt.figure()

    bounded_polar_pspec, bounded_wnum_vals = apply_wnum_bounds(radial_pspec, wnum_vals, wnum_bins,
                                                               (min_lambda, max_lambda))

    dominant_wnum, dominant_theta = find_max(bounded_polar_pspec, bounded_wnum_vals, theta_vals)

    plot_pspec_polar(wnum_bins, theta_bins, radial_pspec, title=my_title)
    plt.scatter(dominant_wnum, dominant_theta, marker='x', color='k', s=100, zorder=100)
    # plt.show()
    plt.figure()

    plot_pspec_polar(wnum_bins, theta_bins,
                     radial_pspec,
                     scale='log', xlim=(0.05, 4.5),
                     vmin=bounded_polar_pspec.min(), vmax=bounded_polar_pspec.max(),
                     title=my_title)
    plt.scatter(dominant_wnum, dominant_theta, marker='x', color='k', s=100, zorder=100)
    plt.tight_layout()
    plt.savefig(save_path + 'polar_pspec.png', dpi=300)
    plt.show()

    print(f'Dominant wavelength: {2 * np.pi / dominant_wnum:.2f} km')
    print(f'Dominant angle: {dominant_theta:.0f} deg from north')

    plot_radial_pspec(radial_pspec, wnum_vals, theta_bins, dominant_wnum, title=my_title)
    plt.savefig(save_path + 'radial_pspec.png', dpi=300)
    # plt.show()

    print('smoothing?')
