import cartopy.crs as ccrs
from pyresample import get_area_def
from satpy import Scene
import numpy as np
import matplotlib.pyplot as plt

filename = 'data/MSG3-SEVI-MSG15-0100-NA-20150414124241.311000000Z-NA/MSG3-SEVI-MSG15-0100-NA-20150414124241.311000000Z-NA.nat'
# load file
global_scene = Scene(reader="seviri_l1b_native", filenames=[filename], reader_kwargs={'fill_disk': True})
global_scene.load(['HRV'], upper_right_corner='NE')
# define area
area_id = '1'
x_size = 1000
y_size = 1000
area_extent = [-11.5, 49.5, 2, 60]
projection = ccrs.PlateCarree().proj4_params
description = "UK"
proj_id = 'PlateCarree'
a = get_area_def(area_id, description, proj_id, projection, x_size, y_size, area_extent)
crs = a.to_cartopy_crs()

scene = global_scene.resample(a)
orig = scene['HRV'].data
ft = np.fft.ifftshift(orig)
ft = np.fft.fft2(ft)
ft = np.fft.fftshift(ft)

plt.imshow(abs(ft), norm= 'log')
plt.show()


