import numpy as np
from file_to_image import produce_scene
import matplotlib.pyplot as plt


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
    xlen = ft.shape[1]
    ylen = ft.shape[0]

    max_k = xlen // 2 * 2 * np.pi / Lx
    max_l = ylen // 2 * 2 * np.pi / Ly

    k = np.linspace(-max_k, max_k, xlen, endpoint=True)
    l = np.linspace(-max_l, max_l, ylen, endpoint=True)

    K, L = np.meshgrid(k, l)

    dist_array = np.sqrt(K ** 2 + L ** 2)

    low_mask = (dist_array < low)
    high_mask = (dist_array > high)
    temp = ft.copy()
    temp[low_mask | high_mask] = 1

    return temp


def basic_plot(img, shifted_ft, Lx, Ly):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    xlen = img.shape[1]
    ylen = img.shape[0]

    pixel_x = Lx / xlen
    pixel_y = Ly / ylen

    ax1.imshow(img,
               extent=[-Lx / 2 - pixel_x / 2, Lx / 2 + pixel_x / 2, -Ly / 2 - pixel_y / 2, Ly / 2 + pixel_y / 2],
               cmap='gray')
    ax1.set_xlabel('x distance / km')
    ax1.set_ylabel('y distance / km')

    max_k = xlen // 2 * 2 * np.pi / Lx
    max_l = ylen // 2 * 2 * np.pi / Ly
    pixel_k = 2 * max_k / xlen
    pixel_l = 2 * max_l / ylen

    ax2.imshow(abs(shifted_ft),
               extent=[-max_k - pixel_k / 2, max_k + pixel_k / 2, -max_l - pixel_l / 2, max_l + pixel_l / 2],
               norm='log')
    ax2.set_xlabel('k / km^-1')
    ax2.set_ylabel('l / km^-1')

    plt.tight_layout()
    plt.show()


def filtered_inv_plot(img, filtered_ft, Lx, Ly):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    xlen = img.shape[1]
    ylen = img.shape[0]

    pixel_x = Lx / xlen
    pixel_y = Ly / ylen

    ax1.imshow(img,
               extent=[-Lx / 2 - pixel_x / 2, Lx / 2 + pixel_x / 2, -Ly / 2 - pixel_y / 2, Ly / 2 + pixel_y / 2],
               cmap='gray')
    ax1.set_xlabel('x distance / km')
    ax1.set_ylabel('y distance / km')

    max_k = xlen // 2 * 2 * np.pi / Lx
    max_l = ylen // 2 * 2 * np.pi / Ly
    pixel_k = 2 * max_k / xlen
    pixel_l = 2 * max_l / ylen

    ax2.imshow(abs(filtered_ft),
               extent=[-max_k - pixel_k / 2, max_k + pixel_k / 2, -max_l - pixel_l / 2, max_l + pixel_l / 2],
               norm='linear')
    ax2.set_xlabel('k / km^-1')
    ax2.set_ylabel('l / km^-1')

    inv = np.fft.ifft2(filtered_ft)
    ax3.imshow(abs(inv),
               extent=[-Lx / 2 - pixel_x / 2, Lx / 2 + pixel_x / 2, -Ly / 2 - pixel_y / 2, Ly / 2 + pixel_y / 2],
               cmap='gray')

    ax3.set_xlabel('x distance / km')
    ax3.set_ylabel('y distance / km')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    filename = 'data/MSG3-SEVI-MSG15-0100-NA-20230419115741.383000000Z-NA/MSG3-SEVI-MSG15-0100-NA-20230419115741.383000000Z-NA.nat'
    area_extent = [-10.5, 51, -5, 55.5]
    Lx = 360
    Ly = 500
    # area_extent = [-11.5, 49.5, 2, 60]
    scene, crs = produce_scene(filename, area_extent=area_extent)
    # img = get_enhanced_image(scene['HRV']).data.transpose('y', 'x', 'bands')

    orig = np.array(scene['HRV'].data)
    ft = np.fft.fft2(orig)
    shifted_ft = np.fft.fftshift(ft)

    # basic_plot(orig, shifted_ft, Lx, Ly)
    min_lambda = 0.1
    max_lambda = 15
    filtered = ideal_bandpass(shifted_ft, Lx, Ly, 2 * np.pi / max_lambda, 2 * np.pi / min_lambda)
    inv = np.fft.ifft2(shifted_ft)
    filtered_inv_plot(orig, filtered, Lx, Ly)
