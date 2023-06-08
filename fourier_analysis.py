import matplotlib.pyplot as plt
import numpy as np
from file_to_image import *
from scipy import ndimage, signal
from miscellaneous import load_settings


# s = load_settings('../tephi_plot/settings/20150414_12_ireland.json')

# filename = 'data/MSG3-SEVI-MSG15-0100-NA-20150414124241.311000000Z-NA/MSG3-SEVI-MSG15-0100-NA-20150414124241' \
#            '.311000000Z-NA.nat'


def basic_plot(x_dist, y_dist):
    fig = plt.figure()
    ax1 = plt.subplot(121, projection=crs)

    ax1.imshow(img, transform=crs, origin='upper', cmap='gray')
    # ax1.imshow(img, transform=crs, origin='upper', extent=[-180, 180, -250, 250], cmap='gray')
    xticks = np.linspace(0, 549, 5)
    xticklabels = [f'{-180 + 360 / 550 * tick:.0f}' for tick in xticks]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)
    yticks = np.linspace(0, 449, 5)
    yticklabels = [f'{-250 + 500 / 450 * tick:.0f}' for tick in yticks]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)

    ax2 = plt.subplot(122)
    # hide ticks and labels
    # ax2.xaxis.set_tick_params(labelbottom=False)
    # ax2.yaxis.set_tick_params(labelleft=False)
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    ax2.imshow(abs(shifted_ft), norm='log')
    plt.tight_layout()
    plt.show()


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

def ideal_bandpass(ft, dist_array, low, high):
    low_mask = (dist_array < low)
    high_mask = (dist_array > high)

    ft[low_mask & high_mask] = 1

    return ft


if __name__ == '__main__':
    filename = 'data/MSG3-SEVI-MSG15-0100-NA-20230419115741.383000000Z-NA/MSG3-SEVI-MSG15-0100-NA-20230419115741.383000000Z-NA.nat'
    area_extent = [-10.5, 51, -5, 55.5]
    # area_extent = [-11.5, 49.5, 2, 60]
    scene, crs = produce_scene(filename, area_extent=area_extent)
    img = get_enhanced_image(scene['HRV']).data.transpose('y', 'x', 'bands')

    orig = scene['HRV'].data
    ft = np.fft.fft2(orig)
    shifted_ft = np.fft.fftshift(ft)

    # low_values_masked_plot()
    # middle_line_masked_plot()
    x_dist = np.linspace(-180, 180, 551, endpoint=True)
    y_dist = np.linspace(-250, 250, 451, endpoint=True)
    x, y = np.meshgrid(x_dist, y_dist)

    dist_array = np.sqrt(x**2 + y**2)
    basic_plot(x_dist, y_dist)





    # plt.hist(img.to_numpy().flatten(), histtype='step', bins=100)
    # plt.show()
#  0.85 km per pixel
