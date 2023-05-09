import matplotlib.pyplot as plt
import numpy as np
from file_to_image import *

filename = 'data/MSG3-SEVI-MSG15-0100-NA-20150414124241.311000000Z-NA/MSG3-SEVI-MSG15-0100-NA-20150414124241' \
           '.311000000Z-NA.nat'
area_extent = [-9.5, 51, -8.5, 52.5]
# area_extent = [-11.5, 49.5, 2, 60]
scene, crs = produce_scene(filename, area_extent=area_extent)
img = get_enhanced_image(scene['HRV']).data.transpose('y', 'x', 'bands')

orig = scene['HRV'].data
# ft = np.fft.ifftshift(orig)
ft = np.fft.fft2(orig)
shifted_ft = np.fft.fftshift(ft)

fig = plt.figure()
ax1 = plt.subplot(121, projection=crs)
ax1.imshow(img, transform=crs, extent=crs.bounds, origin='upper', cmap='gray')

x_size = (area_extent[2] - area_extent[0]) * 100
y_size = (area_extent[3] - area_extent[1]) * 100
# ax1.set_xticks(np.arange(x_size))
# ax1.set_yticks(np.arange(y_size))


ax2 = plt.subplot(122)
# hide ticks and labels
# ax2.xaxis.set_tick_params(labelbottom=False)
# ax2.yaxis.set_tick_params(labelleft=False)
# ax2.set_xticks([])
# ax2.set_yticks([])

ax2.imshow(abs(shifted_ft), norm='log')

plt.tight_layout()
plt.show()

temp = shifted_ft.copy()
temp[temp<200] = 1
plt.imshow(abs(np.fft.ifft2(temp)), cmap='gray')
plt.show()

temp = shifted_ft.copy()
temp[:,50] = 1
plt.imshow(abs(temp))
plt.show()

plt.imshow(abs(np.fft.ifft2(temp)), cmap='gray')
plt.show()
