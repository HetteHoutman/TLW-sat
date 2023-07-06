import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pyproj
from miscellaneous import make_great_circle_points, check_argv_num, load_settings
from pyresample import get_area_def
from satpy import Scene
from satpy.writers import get_enhanced_image

from sonde_locs import sonde_locs


def produce_scene(filename, bottomleft=None, topright=None, grid='latlon'):
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
        # TODO i added '+1' below for testing, should remove
        x_size = Lx // 1000 + 1
        y_size = Ly // 1000 + 1

    projection = ccrs.PlateCarree().proj4_params
    description = "UK"
    proj_id = 'PlateCarree'
    a = get_area_def(area_id, description, proj_id, projection, x_size, y_size, [*bottomleft, *topright])
    crs = a.to_cartopy_crs()

    scene2 = global_scene.resample(a)
    return scene2, crs


def produce_image(scene2, crs, filename, coastlines=False, save_name=None, save=False, great_circle=None):
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
    img = get_enhanced_image(scene2['HRV']).data.transpose('y', 'x', 'bands')
    ax.imshow(img, transform=crs, extent=crs.bounds, origin='upper', cmap='gray')
    if coastlines:
        ax.coastlines()
    # ax.plot([-9.19, -8.82], [51.74, 51.99], marker='x', transform=crs, color='red')
    # ax.plot([-9.66, -9.37], [51.73, 51.99], marker='x', transform=crs, color='blue')

    if great_circle is not None:
        ax.plot(great_circle[0], great_circle[1], color='r', zorder=50)

    plt.scatter(*sonde_locs['valentia'], marker='*', color='r', edgecolors='k', s=250, zorder=100)
    if save:
        if save_name is None:
            save_name = f'{filename[-32:-28]}-{filename[-28:-26]}-{filename[-26:-24]}_{filename[-24:-22]}_{filename[-22:-20]}'
            if coastlines:
                save_name = 'coastlines_' + save_name
        plt.savefig('images/' + save_name + '.png', dpi=300)

    return fig, ax


if __name__ == '__main__':
    # check argument number and load settings
    check_argv_num(sys.argv, 1, "(settings json file)")
    s = load_settings(sys.argv[1])

    gc, dists = make_great_circle_points(s.gc_start, s.gc_end, n=s.n)
    scene, crs = produce_scene(s.sat_file,
                               bottomleft=s.satellite_bottomleft,
                               topright=s.satellite_topright
                               )

    fig, ax = produce_image(scene, crs, s.sat_file, coastlines=True, save=True, save_name='test', great_circle=gc)
    plt.show()
