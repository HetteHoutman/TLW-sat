import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from astropy.convolution import convolve, Gaussian2DKernel
from fourier import *
from fourier_plot import plot_pspec_polar, plot_radial_pspec, plot_2D_pspec, filtered_inv_plot, plot_corr
from miscellaneous import check_argv_num, load_settings, get_region_var
from psd import periodic_smooth_decomp
from skimage.draw import ellipse
from skimage.transform import rotate

from file_to_image import produce_scene


def correlate(a1, b1):
    a = a1.copy()
    b = b1.copy()

    assert a.shape == b.shape

    num = np.sum((a - a.mean()) * (b - b.sum()))

    return num


def correlate_ellipse(pspec, angles, shape):
    """shape has to be longer in y direction"""
    correlation_array = np.zeros_like(pspec.data)

    # loop over elements
    for iy, ix in np.ndindex(pspec.shape):
        # only consider points within lambda-range
        if not pspec.mask[iy, ix]:
            # rotate according to theta
            size = max(shape)
            ell = np.zeros((size, size))
            ell[ellipse(size // 2, size // 2, *shape, shape=ell.shape, rotation=0)] = 1
            ell = rotate(ell, -angles[iy, ix], resize=False)

            # select correct sub-matrix around pixel to correlate with
            half_y_len = ell.shape[0] // 2
            half_x_len = ell.shape[1] // 2
            sub_matrix = pspec.data[iy - half_y_len: iy + half_y_len + 1, ix - half_x_len: ix + half_x_len + 1]

            # correlate and assign to pixel
            correlation_array[iy, ix] = correlate(sub_matrix, ell / ell.sum())

    return np.ma.masked_where(pspec.mask, correlation_array)


if __name__ == '__main__':
    # TODO put things into functions to clean up and make reusable

    k2 = True
    smoothed = True
    mag_filter = False
    test = False

    check_argv_num(sys.argv, 2, "(settings, region json files)")
    s = load_settings(sys.argv[1])
    datetime = f'{s.year}-{s.month:02d}-{s.day:02d}_{s.h}'

    sat_bounds = get_region_var("sat_bounds", sys.argv[2],
                                r"C:/Users/sw825517/OneDrive - University of Reading/research/code/tephi_plot/regions/")
    sat_bl, sat_tr = sat_bounds[:2], sat_bounds[2:]

    if not os.path.exists('plots/' + datetime):
        os.makedirs('plots/' + datetime)

    save_path = f'plots/{datetime}/{sys.argv[2]}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    my_title = f'{datetime}_{sys.argv[2]}_sat'

    if test:
        save_path = f'plots/test/'
        my_title += '_test'

    if k2:
        save_path += 'k2_'
        my_title += '_k2'

    if smoothed:
        save_path += 'smoothed_'
        my_title += '_smoothed'

    if mag_filter:
        save_path += 'magfiltered_'
        my_title += '_magfiltered'

    # produce image
    scene, crs = produce_scene(s.sat_file, bottomleft=sat_bl, topright=sat_tr, grid='km')
    Lx, Ly = extract_distances(scene['HRV'].y[::-1], scene['HRV'].x)
    orig = np.array(scene['HRV'].data)
    # orig -= orig.mean()

    # define reciprocal space
    K, L, wavenumbers, thetas = recip_space(Lx, Ly, orig.shape)
    wavelengths = 2 * np.pi / wavenumbers

    if mag_filter:
        orig[(orig > 12) & (orig < 25)] = 6

    # orig = stripey_test(orig, Lx, Ly, [10], [15], wiggle=20, wiggle_wavelength=20)

    # perform decomposition to remove cross-like signal
    orig, smooth = periodic_smooth_decomp(orig)

    plt.hist(orig.flatten(), bins=100)
    plt.savefig(save_path + 'hist.png', dpi=300)

    ft = np.fft.fft2(orig)
    shifted_ft = np.fft.fftshift(ft)

    min_lambda = 4
    max_lambda = 30
    bandpassed = ideal_bandpass(shifted_ft, Lx, Ly, 2 * np.pi / max_lambda, 2 * np.pi / min_lambda)
    plt.figure()
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
        pspec_2d = np.ma.masked_where(pspec_2d.mask, convolve(pspec_2d.data, Gaussian2DKernel(7, x_size=15, y_size=15),
                                                              boundary='wrap'))
        radial_pspec = convolve(radial_pspec, Gaussian2DKernel(3, x_size=11, y_size=11), boundary='wrap')

    plot_2D_pspec(pspec_2d, Lx, Ly, wavelength_contours=[5, 10, 35], title=my_title)
    plt.savefig(save_path + '2d_pspec.png', dpi=300)
    # plt.show()
    plt.figure()

    bounded_polar_pspec, bounded_wnum_vals = apply_wnum_bounds(radial_pspec, wnum_vals, wnum_bins,
                                                               (min_lambda, max_lambda))

    dominant_wnum, dominant_theta = find_max(bounded_polar_pspec, bounded_wnum_vals, theta_vals)

    plot_pspec_polar(wnum_bins, theta_bins,
                     radial_pspec,
                     scale='log', xlim=(0.05, 4.5),
                     vmin=np.nanmin(bounded_polar_pspec), vmax=np.nanmax(bounded_polar_pspec),
                     title=my_title, min_lambda=min_lambda, max_lambda=max_lambda)
    plt.scatter(dominant_wnum, dominant_theta, marker='x', color='k', s=100, zorder=100)
    plt.tight_layout()
    plt.savefig(save_path + 'polar_pspec.png', dpi=300)
    # plt.show()
    plt.figure()

    print(f'Dominant wavelength: {2 * np.pi / dominant_wnum:.2f} km')
    print(f'Dominant angle: {dominant_theta:.0f} deg from north')

    plot_radial_pspec(radial_pspec, wnum_vals, theta_bins, dominant_wnum, title=my_title)
    plt.savefig(save_path + 'radial_pspec.png', dpi=300)
    # plt.show()
    plt.figure()

    np.save(f'data/{my_title}', bounded_polar_pspec)

    corr = correlate_ellipse(pspec_2d, thetas, (2, 25))
    # num, den = correlate_ellipse(pspec_2d, thetas, (2, 30))
    #
    # plt.imshow(num)
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(den)
    # plt.colorbar()
    # plt.show()
    #
    # corr = num / den

    rot_left_half = rotate(corr[:, :corr.shape[1] // 2 + 1], 180)
    collapsed_corr = corr[:, corr.shape[1] // 2:] + rot_left_half

    # mask bottom half of k_x = 0 line as this is the same as the top half
    collapsed_corr.mask[collapsed_corr.shape[0] // 2:, 0] = True

    idxs = np.unravel_index(collapsed_corr.argmax(), collapsed_corr.shape)
    dom_K, dom_L = K[:, K.shape[1] // 2:][*idxs], L[::-1, L.shape[1] // 2:][*idxs]
    dominant_wlen = wavelengths[:, wavelengths.shape[1] // 2:][*idxs]
    dominant_theta = thetas[:, thetas.shape[1] // 2:][*idxs]

    plot_corr(collapsed_corr, L, K)
    plt.scatter(dom_K, dom_L, marker='x')
    plt.savefig(save_path + 'corr.png', dpi=300)
    plt.show()
    plt.figure()

    plot_2D_pspec(pspec_2d, Lx, Ly, wavelength_contours=[5, 10, 35], title=my_title)
    plt.scatter(dom_K, dom_L, marker='x')
    plt.savefig(save_path + '2d_pspec_withcross.png', dpi=300)
    plt.show()

    print(f'Dominant wavelength by ellipse method: {dominant_wlen:.2f} km')
    print(f'Dominant angle by ellipse method: {dominant_theta:.0f} deg from north')

    df = pd.read_excel('../../other_data/sat_vs_ukv_results.xlsx', index_col=[0, 1])
    df.loc[(f'{s.year}-{s.month:02d}-{s.day:02d}', sys.argv[2]), 'sat_lambda_ellipse'] = dominant_wlen
    df.loc[(f'{s.year}-{s.month:02d}-{s.day:02d}', sys.argv[2]), 'sat_theta_ellipse'] = dominant_theta
    df.to_excel('../../other_data/sat_vs_ukv_results.xlsx')

