import numpy as np
from scipy import stats

from file_to_image import produce_scene
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyproj


# s = load_settings('../tephi_plot/settings/20150414_12_ireland.json')

# filename = 'data/MSG3-SEVI-MSG15-0100-NA-20150414124241.311000000Z-NA/MSG3-SEVI-MSG15-0100-NA-20150414124241' \
#            '.311000000Z-NA.nat'

def low_values_masked_plot():
    global fig, axes, temp
    fig, axes = plt.subplots(1, 2)
    temp = shifted_ft.copy()
    temp[temp < 100] = 1
    axes[0].imshow(abs(temp), norm='log')
    axes[1].imshow(abs(np.fft.ifft2(temp)), cmap='gray')
    plt.show()


def middle_line_masked_plot():
    global fig, axes, temp
    fig, axes = plt.subplots(1, 2)
    temp = shifted_ft.copy()
    temp[:, temp.shape[1] // 2] = 1
    # temp[temp.shape[0] // 2] = 1
    axes[0].imshow(abs(temp), norm='log')
    axes[1].imshow(abs(np.fft.ifft2(temp)), cmap='gray')
    plt.show()


def ideal_bandpass(ft, Lx, Ly, low, high):
    _, _, dist_array, thetas = recip_distances(Lx, Ly, ft)

    low_mask = (dist_array < low)
    high_mask = (dist_array > high)
    masked = np.ma.masked_where(low_mask | high_mask, ft)

    return masked


def low_values_mask(ft, threshold):
    masked = np.ma.masked_where(abs(ft) < threshold, ft)
    return masked


def recip_distances(Lx, Ly, ft):
    # TODO use shape instead of ft
    xlen = ft.shape[1]
    ylen = ft.shape[0]

    max_k = xlen // 2 * 2 * np.pi / Lx
    max_l = ylen // 2 * 2 * np.pi / Ly

    k = np.linspace(-max_k, max_k, xlen, endpoint=True)
    l = np.linspace(-max_l, max_l, ylen, endpoint=True)
    K, L = np.meshgrid(k, l)

    dist_array = np.sqrt(K ** 2 + L ** 2)
    thetas = np.rad2deg(np.arctan2(K, L)) + 90
    thetas %= 360
    return K, L, dist_array, thetas


def filtered_inv_plot(img, filtered_ft, Lx, Ly, latlon=None, inverse_fft=True, wavelength=True):
    # TODO want to be able to plot lambda instead of k/l

    if inverse_fft:
        fig, (ax1, ax3) = plt.subplots(1, 2, sharey=True)
    else:
        fig, ax1 = plt.subplots(1, 1)

    xlen = img.shape[1]
    ylen = img.shape[0]

    if latlon:
        physical_extent = [latlon[0], latlon[2], latlon[1], latlon[3]]
        xlabel = 'Longitude'
        ylabel = 'Latitude'
    else:
        pixel_x = Lx / xlen
        pixel_y = Ly / ylen
        physical_extent = [-Lx / 2 - pixel_x / 2, Lx / 2 + pixel_x / 2, -Ly / 2 - pixel_y / 2, Ly / 2 + pixel_y / 2]
        xlabel = 'x distance / km'
        ylabel = 'y distance / km'

    ax1.imshow(img,
               extent=physical_extent,
               cmap='gray')
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)

    if inverse_fft:
        inv = np.fft.ifft2(filtered_ft.filled(fill_value=1))
        ax3.imshow(abs(inv),
                   extent=physical_extent,
                   cmap='gray')
    # save?
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(1, 1)

    max_k = xlen // 2 * 2 * np.pi / Lx
    max_l = ylen // 2 * 2 * np.pi / Ly
    pixel_k = 2 * max_k / xlen
    pixel_l = 2 * max_l / ylen
    recip_extent = [-max_k - pixel_k / 2, max_k + pixel_k / 2, -max_l - pixel_l / 2, max_l + pixel_l / 2]

    im = ax2.imshow(abs(filtered_ft)**2,
               extent=recip_extent,
               norm='log')
    if wavelength:
        K, L, dist_array, thetas = recip_distances(Lx, Ly, ft)
        wavelengths = 2 * np.pi / dist_array
        con = ax2.contour(K, L, wavelengths, levels=[5, 10], colors=['k'], linestyles=['--'])
        ax2.clabel(con)

    ax2.set_title('2D Power Spectrum')
    ax2.set_xlabel('k / km^-1')
    ax2.set_ylabel('l / km^-1')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    fig2.colorbar(im)

    plt.show()


if __name__ == '__main__':
    filename = 'data/MSG3-SEVI-MSG15-0100-NA-20230419115741.383000000Z-NA/MSG3-SEVI-MSG15-0100-NA-20230419115741.383000000Z-NA.nat'
    area_extent = [-9, 54, -8, 55]
    # area_extent = [-11.5, 49.5, 2, 60]
    g = pyproj.Geod(ellps='WGS84')
    _, _, Lx = g.inv(area_extent[0], (area_extent[3] + area_extent[1]) / 2,
                     area_extent[2], (area_extent[3] + area_extent[1]) / 2)
    _, _, Ly = g.inv((area_extent[0] + area_extent[2]) / 2, area_extent[1],
                     (area_extent[0] + area_extent[2]) / 2, area_extent[3])
    Lx /= 1000
    Ly /= 1000
    scene, crs = produce_scene(filename, area_extent=area_extent)

    orig = np.array(scene['HRV'].data)
    ft = np.fft.fft2(orig)
    shifted_ft = np.fft.fftshift(ft)

    # basic_plot(orig, shifted_ft, Lx, Ly)
    min_lambda = 5
    max_lambda = 35
    bandpassed = ideal_bandpass(shifted_ft, Lx, Ly, 2 * np.pi / max_lambda, 2 * np.pi / min_lambda)
    threshold = abs(bandpassed.compressed()).mean()
    filtered = low_values_mask(bandpassed, threshold)
    filtered_inv_plot(orig, bandpassed, Lx, Ly,
                      latlon=area_extent,
                      inverse_fft=True)

    # -------- other stuff -------------
    K, L, wavenumbers, thetas = recip_distances(Lx, Ly, ft)
    wavelengths = 2 * np.pi / wavenumbers
    selec_dists = wavenumbers[~filtered.mask]
    highest = 115
    indices = np.argpartition(-abs(filtered.compressed()), highest)[:highest]
    print(
        f'Mean wavelength of the highest (most dominant) {highest} wavelengths is: {np.mean(2 * np.pi / selec_dists[indices]):.2f} km')
    print(f'Average wavelength with fft value above the mean fft value weighted by fft value is '
          f'{np.average(wavelengths[~filtered.mask], weights=abs(filtered).compressed()):.2f} km')

    # -------- power spectrum ----------
    filtered_amplitudes = abs(bandpassed) ** 2

    kbins = np.arange(1, 60) / 10
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    pspec, _, _ = stats.binned_statistic(wavenumbers.flatten(), filtered_amplitudes.flatten(),
                                         statistic="mean",
                                         bins=kbins)
    pspec *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

    theta_parts = 144
    thetabins = np.linspace(0, 360, theta_parts, endpoint=True)
    thetavals = 0.5 * (thetabins[1:] + thetabins[:-1])
    pspec_ang, _, _ = stats.binned_statistic(thetas.flatten(), filtered_amplitudes.flatten(),
                                         statistic="mean",
                                         bins=thetabins)
    pspec_ang *= np.pi * ((2 * np.pi / min_lambda)**2 - (2 * np.pi / max_lambda)**2) / theta_parts

    plt.loglog(kvals, pspec)

    xlen = orig.shape[1]
    ylen = orig.shape[0]
    pixel_x = Lx / xlen
    pixel_y = Ly / ylen
    ymin = np.nanmin(pspec)
    ymax = np.nanmax(pspec)
    plt.vlines(2 * np.pi / 8, ymin, ymax, 'k', linestyles='--')
    plt.vlines(2 * np.pi / min(Lx, Ly), ymin, ymax, 'k', linestyles='dotted')
    plt.vlines(np.pi / max(pixel_y, pixel_x), ymin, ymax, 'k', linestyles='dotted')
    plt.vlines(2 * np.pi / min_lambda, ymin, ymax, 'k', linestyles='-.')
    plt.vlines(2 * np.pi / max_lambda, ymin, ymax, 'k', linestyles='-.')

    plt.title('1D Power Spectrum')
    plt.xlabel("k / km^-1")
    plt.ylabel("$P(k)$")
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.show()

    plt.plot(thetavals, pspec_ang)
    ax = plt.gca()
    ax.set_yscale('log')
    plt.title('Angular power spectrum')

    plt.ylabel("$P(k)$")
    plt.xlabel(r'$\theta$ (deg)')
    plt.grid()
    plt.show()