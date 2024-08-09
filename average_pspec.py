import sys

import pandas as pd
import numpy as np

from fourier import recip_space, make_polar_pspec
from wavelet_analysis import get_seviri_img

df = pd.read_excel(sys.argv[1], parse_dates=[0])
regions = ['scotland', 'ireland', 'england', 'wales', 'cornwall', 'north_england']
df = df[df.dates.dt.year==2023]

wnum_bin_width = 0.025
theta_bin_width = 2.5
region = 'scotland'

ps = np.zeros((len(df), 178))
norm_ps = np.zeros((len(df), 178))
means = np.zeros(len(df))
stds = np.zeros(len(df))

for i, (date, hour) in enumerate(zip(df.dates, df.h)):
    datetime = date.replace(hour=int(hour))
    print(datetime)
    orig, Lx, Ly = get_seviri_img(datetime, region)
    # orig -= orig.mean()
    # orig /= orig.std()
    K, L, wavenumbers, thetas = recip_space(Lx, Ly, orig.shape)
    ft = np.fft.fftshift(np.fft.fft2(orig))
    p = abs(ft) ** 2
    polar_p, wnum_bins, wnum_vals, theta_bins, theta_vals = make_polar_pspec(p, wavenumbers, wnum_bin_width,
                                                                                  thetas, theta_bin_width)

    ps[i] = np.nanmean(polar_p, axis=0)
    norm_ps[i] = np.nanmean(polar_p, axis=0) / orig.size / orig.var()
    means[i] = orig.mean()
    stds[i] = orig.std()

# radial_p /= len(df.dates)
# mean_p /= len(df.dates)
# np.save('scotland_mean_pspec.npy', mean_p)