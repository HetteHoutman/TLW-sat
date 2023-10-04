import os
import sys

import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian2DKernel
from fourier import *
from fourier_plot import plot_pspec_polar, plot_radial_pspec, plot_2D_pspec, filtered_inv_plot
from miscellaneous import check_argv_num, load_settings, get_region_var
from psd import periodic_smooth_decomp
from skimage.morphology import ellipse
from skimage.transform import rotate

from file_to_image import produce_scene


def correlate(a1, b1):
    a = a1.copy()
    b = b1.copy()
    a -= a.mean()
    b -= b.mean()

    assert a.shape == b.shape

    temp = a * b
    norm = np.sqrt(np.sum(a * a) * np.sum(b * b))
    norm = 1
    return np.sum(temp) / norm


def correlate_ellipse(pspec, angles, shape):
    correlation_array = np.zeros_like(pspec.data)
    # loop over elements
    for iy, ix in np.ndindex(pspec.shape):
        # only consider points within lambda-range
        if not pspec.mask[iy, ix]:
            # rotate according to theta
            ell = rotate(ellipse(*shape), 90 - angles[iy, ix], resize=True)
            # make ellipse shape odd so that it has a middle pixel
            if ell.shape[0] % 2 == 0:
                ell = ell[1:]
            if ell.shape[1] % 2 == 0:
                ell = ell[:, 1:]

            half_y_len = ell.shape[0] // 2
            half_x_len = ell.shape[1] // 2
            sub_matrix = pspec.data[iy - half_y_len: iy + half_y_len + 1, ix - half_x_len: ix + half_x_len + 1]
            correlation_array[iy, ix] = correlate(sub_matrix, ell / ell.sum())

            if iy == 150 and ix == 105:
                henk = pspec.data.copy() / pspec.data.max()
                henk[iy - half_y_len: iy + half_y_len + 1, ix - half_x_len: ix + half_x_len + 1] += (
                        ell / ell.shape[0] / ell.shape[1])
                plt.imshow(henk)
                plt.show()

    return np.ma.masked_where(pspec.mask, correlation_array)


if __name__ == '__main__':
    k2 = True
    smoothed = True
    mag_filter = False
    test = False

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
    orig, s = periodic_smooth_decomp(orig)

    plt.hist(orig.flatten(), bins=100)
    plt.savefig(save_path + 'hist.png', dpi=300)

    ft = np.fft.fft2(orig)
    shifted_ft = np.fft.fftshift(ft)

    min_lambda = 4
    max_lambda = 35
    bandpassed = ideal_bandpass(shifted_ft, Lx, Ly, 2 * np.pi / max_lambda, 2 * np.pi / min_lambda)
    plt.figure()
    filtered_inv_plot(orig, bandpassed, Lx, Ly, inverse_fft=True, title=my_title
                      # latlon=area_extent
                      )
    plt.savefig(save_path + 'sat_plot.png', dpi=300)
    plt.show()
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
    plt.show()
    plt.figure()

    bounded_polar_pspec, bounded_wnum_vals = apply_wnum_bounds(radial_pspec, wnum_vals, wnum_bins,
                                                               (min_lambda, max_lambda))

    dominant_wnum, dominant_theta = find_max(bounded_polar_pspec, bounded_wnum_vals, theta_vals)

    plot_pspec_polar(wnum_bins, theta_bins, radial_pspec, title=my_title, min_lambda=min_lambda, max_lambda=max_lambda)
    plt.scatter(dominant_wnum, dominant_theta, marker='x', color='k', s=100, zorder=100)
    # plt.show()
    plt.figure()

    plot_pspec_polar(wnum_bins, theta_bins,
                     radial_pspec,
                     scale='log', xlim=(0.05, 4.5),
                     vmin=np.nanmin(bounded_polar_pspec), vmax=np.nanmax(bounded_polar_pspec),
                     title=my_title, min_lambda=min_lambda, max_lambda=max_lambda)
    plt.scatter(dominant_wnum, dominant_theta, marker='x', color='k', s=100, zorder=100)
    plt.tight_layout()
    plt.savefig(save_path + 'polar_pspec.png', dpi=300)
    plt.show()

    print(f'Dominant wavelength: {2 * np.pi / dominant_wnum:.2f} km')
    print(f'Dominant angle: {dominant_theta:.0f} deg from north')

    plot_radial_pspec(radial_pspec, wnum_vals, theta_bins, dominant_wnum, title=my_title)
    plt.savefig(save_path + 'radial_pspec.png', dpi=300)
    # plt.show()
    plt.figure()

    np.save(f'data/{my_title}', bounded_polar_pspec)

    corr = correlate_ellipse(pspec_2d, thetas, (2, 30))

    rot_left_half = rotate(corr[:, :corr.shape[1] // 2 + 1], 180)
    collapsed_corr = corr[:, corr.shape[1] // 2:] + rot_left_half

    # mask bottom half of k_x = 0 line as this is the same as the top half
    collapsed_corr.mask[collapsed_corr.shape[0] // 2:, 0] = True

    idxs = np.unravel_index(collapsed_corr.argmax(), collapsed_corr.shape)

    plt.imshow(collapsed_corr)
    plt.scatter(idxs[1], idxs[0], marker='x')
    plt.show()

    plot_2D_pspec(pspec_2d, Lx, Ly, wavelength_contours=[5, 10, 35], title=my_title)
    plt.scatter(K[:, K.shape[1] // 2:][*idxs], L[::-1, L.shape[1] // 2:][*idxs], marker='x')
    plt.show()

    print(f'Dominant wavelength by ellispe method: {wavelengths[:, wavelengths.shape[1] // 2:][*idxs]:.2f} km')
    print(f'Dominant angle: {thetas[:, thetas.shape[1] // 2:][*idxs]:.0f} deg from north')
