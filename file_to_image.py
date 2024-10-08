import os.path
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pyproj
from miscellaneous import make_great_circle_points, check_argv_num, load_settings
from prepare_metadata import get_variable_from_region_json
from pyresample import get_area_def
from satpy import Scene
from satpy.writers import get_enhanced_image


def produce_scene(filename, bottomleft=None, topright=None, grid='latlon', pixels_per_km=1):
    if bottomleft is None:
        bottomleft = [-11.5, 49.5]

    if topright is None:
        topright = [2, 60]

    # load file
    global_scene = Scene(reader="seviri_l1b_native", filenames=[filename], reader_kwargs={'fill_disk': True})
    global_scene.load(['HRV'], upper_right_corner='NE')

    # define area
    area_id = '1'
    if grid == 'latlon':
        # 101x101 with constant lat / constant lon pixels (but lat and lon are not necessarily the same)
        x_size = (topright[0] - bottomleft[0]) * 101
        y_size = (topright[1] - bottomleft[1]) * 101

    if grid == 'km':
        # currently only supports pixels of 1 km
        midx = (bottomleft[0] + topright[0]) / 2
        midy = (bottomleft[1] + topright[1]) / 2
        g = pyproj.Geod(ellps='WGS84')
        _, _, Lx = g.inv(bottomleft[0], midy, topright[0], midy)
        _, _, Ly = g.inv(midx, bottomleft[1], midx, topright[1])

        # TODO need to add clause for when x or y is exactly an integer (although might be unlikely...)

        x_size = Lx / 1000 * pixels_per_km
        y_size = Ly / 1000 * pixels_per_km

        # ensure sizes are odd
        if np.floor(x_size) % 2 == 1:
            x_size = np.floor(x_size)
        else:
            x_size = np.ceil(x_size)

        if np.floor(y_size) % 2 == 1:
            y_size = np.floor(y_size)
        else:
            y_size = np.ceil(y_size)

    projection = ccrs.PlateCarree().proj4_params
    description = "UK"
    proj_id = 'PlateCarree'
    a = get_area_def(area_id, description, proj_id, projection, x_size, y_size, [*bottomleft, *topright])
    crs = a.to_cartopy_crs()

    scene2 = global_scene.resample(a)
    return scene2, a


def produce_image(scene2, crs, coastlines=False, save_name=None, save=False, great_circle=None):
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
    img = get_enhanced_image(scene2['HRV']).data.transpose('y', 'x', 'bands')
    ax.imshow(img, transform=crs, extent=crs.bounds, origin='upper', cmap='gray')
    if coastlines:
        ax.coastlines()
    # ax.plot([-9.19, -8.82], [51.74, 51.99], marker='x', transform=crs, color='red')
    # ax.plot([-9.66, -9.37], [51.73, 51.99], marker='x', transform=crs, color='blue')

    if great_circle is not None:
        ax.plot(great_circle[0], great_circle[1], color='r', zorder=50)

    # plt.scatter(*sonde_locs['valentia'], marker='*', color='r', edgecolors='k', s=250, zorder=100)

    if save:
        datetime = f'{s.year}-{s.month}-{s.day}_{s.h}'

        if not os.path.exists(f'images/{datetime}'):
            os.makedirs(f'images/{datetime}')

        if save_name is None:
            save_name = sys.argv[2]
            if coastlines:
                save_name = save_name + '_coastlines'
            if gc is not None:
                save_name = save_name + '_gc'

        plt.savefig(f'images/{datetime}/{save_name}.png', dpi=300)

    return fig, ax


if __name__ == '__main__':
    # TODO coastlines seem slightly off?
    # check argument number and load settings
    check_argv_num(sys.argv, 2, "(settings, region json files)")
    s = load_settings(sys.argv[1])
    sat_bounds = get_variable_from_region_json("sat_bounds", sys.argv[2],
                                r"C:/Users/sw825517/OneDrive - University of Reading/research/code/tephi_plot/regions/")

    sat_bl, sat_tr = sat_bounds[:2], sat_bounds[2:]

    gc = None
    try:
        gc, dists = make_great_circle_points(s.gc_start, s.gc_end, n=s.n)
    except:
        pass

    scene, crs = produce_scene(s.sat_file,
                               bottomleft=sat_bl,
                               topright=sat_tr
                               )

    fig, ax = produce_image(scene, crs, coastlines=False, save=True, save_name=None, great_circle=gc)
    plt.show()
