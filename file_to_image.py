from satpy import Scene
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from satpy.writers import get_enhanced_image
from pyresample import get_area_def


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


def produce_image(scene2, crs, filename, coastlines=False, save_name=None, save=False):
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
    img = get_enhanced_image(scene2['HRV']).data.transpose('y', 'x', 'bands')
    ax.imshow(img, transform=crs, extent=crs.bounds, origin='upper', cmap='gray')
    if coastlines:
        ax.coastlines()
    ax.plot([-9.19, -8.82], [51.74, 51.99], marker='x', transform=crs, color='red')
    ax.plot([-9.66, -9.37], [51.73, 51.99], marker='x', transform=crs, color='blue')
    if save:
        if save_name is None:
            save_name = f'{filename[-32:-28]}-{filename[-28:-26]}-{filename[-26:-24]}_{filename[-24:-22]}_{filename[-22:-20]}'
            if coastlines:
                save_name = 'coastlines_' + save_name
        plt.savefig('images/' + save_name + '.png', dpi=300)
    return fig, ax


if __name__ == '__main__':
    file = 'data/MSG3-SEVI-MSG15-0100-NA-20150414124241.311000000Z-NA/MSG3-SEVI-MSG15-0100-NA-20150414124241' \
           '.311000000Z-NA.nat'
    scene, crs = produce_scene(file,
                               # area_extent=[-11, 50, -7, 53]
                               )

    fig, ax = produce_image(scene, crs, file, save=True, save_name='test')
    plt.show()
