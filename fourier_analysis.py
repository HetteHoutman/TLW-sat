import matplotlib.pyplot as plt
import numpy as np
import pyproj
from scipy import stats

from file_to_image import produce_scene


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
    _, _, dist_array, thetas = recip_space(Lx, Ly, ft.shape)

    low_mask = (dist_array < low)
    high_mask = (dist_array > high)
    masked = np.ma.masked_where(low_mask | high_mask, ft)

    return masked


def low_values_mask(ft, threshold):
    masked = np.ma.masked_where(abs(ft) < threshold, ft)
    return masked


def recip_space(Lx, Ly, shape):
    xlen = shape[1]
    ylen = shape[0]

    # np uses linear frequency f instead of angular frequency omega=2pi*f, so multiply by 2pi to get angular wavenum k
    k = 2 * np.pi * np.fft.fftfreq(xlen, d=Lx / xlen)
    l = 2 * np.pi * np.fft.fftfreq(ylen, d=Ly / ylen)

    # do fft shift
    K, L = np.meshgrid(np.roll(k, k.shape[0] // 2), np.roll(l, l.shape[0] // 2))

    dist_array = np.sqrt(K ** 2 + L ** 2)
    thetas = -np.rad2deg(np.arctan2(K, L)) + 180
    thetas %= 180
    return K, L, dist_array, thetas


def extract_distances(lats, lons):
    g = pyproj.Geod(ellps='WGS84')
    _, _, Lx = g.inv(lons[0], lats[lats.shape[0] // 2],
                     lons[-1], lats[lats.shape[0] // 2])
    _, _, Ly = g.inv(lons[lons.shape[0] // 2], lats[0],
                     lons[lons.shape[0] // 2], lats[-1])

    return Lx / 1000, Ly / 1000


def create_bins(range, bin_width):
    bins = np.linspace(range[0], np.ceil(range[1]), int(range[1] / bin_width) + 1)
    vals = 0.5 * (bins[1:] + bins[:-1])
    return bins, vals


def filtered_inv_plot(img, filtered_ft, Lx, Ly, latlon=None, inverse_fft=True):

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
        ax3.set_title(f'{min_lambda} km < lambda < {max_lambda} km')
        ax3.imshow(abs(inv),
                   extent=physical_extent,
                   cmap='gray')
    # save?
    plt.tight_layout()
    plt.show()


def plot_2D_pspec(pspec, Lx, Ly, wavelength_contours=None):
    xlen = pspec.shape[1]
    ylen = pspec.shape[0]

    fig2, ax2 = plt.subplots(1, 1)
    # TODO change to pcolormesh?
    max_k = xlen // 2 * 2 * np.pi / Lx
    max_l = ylen // 2 * 2 * np.pi / Ly
    pixel_k = 2 * max_k / xlen
    pixel_l = 2 * max_l / ylen
    recip_extent = [-max_k - pixel_k / 2, max_k + pixel_k / 2, -max_l - pixel_l / 2, max_l + pixel_l / 2]

    im = ax2.imshow(pspec.data, extent=recip_extent, interpolation='none',
                    norm='log', vmin=pspec.min(), vmax=pspec.max())

    if wavelength_contours:
        K, L, dist_array, thetas = recip_space(Lx, Ly, pspec.shape)
        wavelengths = 2 * np.pi / dist_array
        con = ax2.contour(K, L, wavelengths, levels=wavelength_contours, colors=['k'], linestyles=['--'])
        ax2.clabel(con)

    ax2.set_title('2D Power Spectrum')
    ax2.set_xlabel('k / km^-1')
    ax2.set_ylabel('l / km^-1')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    fig2.colorbar(im, extend='both')
    plt.tight_layout()
    plt.show()


def plot_radial_pspec(pspec, vals):
    plt.loglog(vals, pspec)

    xlen = orig.shape[1]
    ylen = orig.shape[0]
    pixel_x = Lx / xlen
    pixel_y = Ly / ylen
    ymin = np.nanmin(radial_pspec)
    ymax = np.nanmax(radial_pspec)

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


def plot_ang_pspec(pspec, vals):
    plt.plot(vals, pspec)
    ax = plt.gca()
    ax.set_yscale('log')
    plt.title('Angular power spectrum')
    plt.ylabel(r"$P(\theta)$")
    plt.xlabel(r'$\theta$ (deg)')
    plt.grid()
    plt.show()


def make_stripes(X, Y, wavelength, angle):
    angle += 90
    angle = np.deg2rad(angle)
    return np.sin(2 * np.pi * (X * np.cos(angle) + Y * np.sin(angle)) / wavelength)


if __name__ == '__main__':
    filename = 'data/MSG3-SEVI-MSG15-0100-NA-20230419115741.383000000Z-NA/MSG3-SEVI-MSG15-0100-NA-20230419115741.383000000Z-NA.nat'
    area_extent = [-9, 54, -8, 55]
    # area_extent = [-11.5, 49.5, 2, 60]
    scene, crs = produce_scene(filename, area_extent=area_extent)
    Lx, Ly = extract_distances(scene['HRV'].y[::-1], scene['HRV'].x)
    orig = np.array(scene['HRV'].data)

    x = np.linspace(-Lx / 2, Lx / 2, orig.shape[1])
    y = np.linspace(-Ly / 2, Ly / 2, orig.shape[0])
    X, Y = np.meshgrid(x, y)
    stripe1 = make_stripes(X, Y, 10, 45)
    stripe2 = make_stripes(X, Y, 5, 135)

    orig = stripe1 + stripe2

    ft = np.fft.fft2(orig)
    shifted_ft = np.fft.fftshift(ft)

    # basic_plot(orig, shifted_ft, Lx, Ly)
    min_lambda = 5
    max_lambda = 35
    bandpassed = ideal_bandpass(shifted_ft, Lx, Ly, 2 * np.pi / max_lambda, 2 * np.pi / min_lambda)
    filtered_inv_plot(orig, bandpassed, Lx, Ly,
                      # latlon=area_extent,
                      inverse_fft=True)
    pspec_2d = np.ma.masked_where(bandpassed.mask, abs(shifted_ft) ** 2)
    plot_2D_pspec(pspec_2d, Lx, Ly, wavelength_contours=[5, 10, 35])
    K, L, wavenumbers, thetas = recip_space(Lx, Ly, ft.shape)

    # -------- power spectrum ----------
    # TODO check if this is mathematically the right way of calculating pspec
    wnum_bin_width = 0.1
    wnum_bins, wnum_vals = create_bins((0, wavenumbers.max()), wnum_bin_width)
    radial_pspec, _, _ = stats.binned_statistic(wavenumbers.flatten(), pspec_2d.data.flatten(),
                                                statistic="mean",
                                                bins=wnum_bins)
    radial_pspec *= np.pi * (wnum_bins[1:] ** 2 - wnum_bins[:-1] ** 2)

    theta_bin_width = 10
    theta_bins, theta_vals = create_bins((0, 180), theta_bin_width)
    ang_pspec, _, _ = stats.binned_statistic(thetas[~pspec_2d.mask], pspec_2d.compressed(),
                                             statistic="mean",
                                             bins=theta_bins)
    ang_pspec *= np.deg2rad(theta_bin_width) * ((2 * np.pi / min_lambda) ** 2 - (2 * np.pi / max_lambda) ** 2)

    plot_radial_pspec(radial_pspec, wnum_vals)

    plot_ang_pspec(ang_pspec, theta_vals)
