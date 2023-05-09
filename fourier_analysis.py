import cartopy.crs as ccrs
from pyresample import get_area_def
from satpy import Scene
import numpy as np
import matplotlib.pyplot as plt
from file_to_image import produce_scene

filename = 'data/MSG3-SEVI-MSG15-0100-NA-20150414124241.311000000Z-NA/MSG3-SEVI-MSG15-0100-NA-20150414124241' \
           '.311000000Z-NA.nat'
scene, crs = produce_scene(filename)

orig = scene['HRV'].data
ft = np.fft.ifftshift(orig)
ft = np.fft.fft2(ft)
ft = np.fft.fftshift(ft)

plt.imshow(abs(ft), norm= 'log')
plt.show()


