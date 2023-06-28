import numpy as np
import pyproj


def ideal_bandpass(ft, Lx, Ly, low, high):
    _, _, dist_array, thetas = recip_space(Lx, Ly, ft.shape)

    low_mask = (dist_array < low)
    high_mask = (dist_array > high)
    masked = np.ma.masked_where(low_mask | high_mask, ft)

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
    bins = np.linspace(range[0], range[1], int(np.ceil(range[1] / bin_width) + 1))
    vals = 0.5 * (bins[1:] + bins[:-1])
    return bins, vals