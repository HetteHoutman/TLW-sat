import json
import sys
from types import SimpleNamespace

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from pyresample import get_area_def
from satpy import Scene
from satpy.writers import get_enhanced_image

from miscellaneous import make_great_circle_points

def produce_scene(filename, area_extent=None):
    # load file
    if area_extent is None:
        area_extent = [-11.5, 49.5, 2, 60]
    global_scene = Scene(reader="seviri_l1b_native", filenames=[filename], reader_kwargs={'fill_disk': True})
    global_scene.load(['HRV'], upper_right_corner='NE')

    # define area
    area_id = '1'
    x_size = (area_extent[2] - area_extent[0]) * 100
    y_size = (area_extent[3] - area_extent[1]) * 100
    projection = ccrs.PlateCarree().proj4_params
    description = "UK"
    proj_id = 'PlateCarree'
    a = get_area_def(area_id, description, proj_id, projection, x_size, y_size, area_extent)
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
    
    if save:
        if save_name is None:
            save_name = f'{filename[-32:-28]}-{filename[-28:-26]}-{filename[-26:-24]}_{filename[-24:-22]}_{filename[-22:-20]}'
            if coastlines:
                save_name = 'coastlines_' + save_name
        plt.savefig('images/' + save_name + '.png', dpi=300)
    return fig, ax

def load_settings():
    """borrowed from tephi_plot"""
    if len(sys.argv) != 2:
        raise Exception(f'Gave {len(sys.argv) - 1} arguments but this file takes exactly 1 (settings.json)')
    file = sys.argv[1]
    with open(file) as f:
        settings = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    return settings


if __name__ == '__main__':
    s = load_settings()
    # TODO include satellite file in settings json
    file = 'data/MSG3-SEVI-MSG15-0100-NA-20150414122741.433000000Z-NA/MSG3-SEVI-MSG15-0100-NA-20150414122741.433000000Z-NA.nat'
    gc, dists = make_great_circle_points(s.gc_start, s.gc_end, n=s.n)
    scene, crs = produce_scene(file,
                               area_extent=[*s.map_bottomleft, *s.map_topright]
                               )
    fig, ax = produce_image(scene, crs, file, coastlines=True, save=True, save_name='test', great_circle=gc)
    plt.show()
