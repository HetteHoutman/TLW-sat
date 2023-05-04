from satpy import Scene
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from satpy.writers import get_enhanced_image
from pyresample import get_area_def


def produce_image(filename, save=True, save_name=None, coastlines=False):
    # load file
    global_scene = Scene(reader="seviri_l1b_native", filenames=[filename], reader_kwargs={'fill_disk': True})
    global_scene.load(['HRV'], upper_right_corner='NE')
    print(global_scene.available_dataset_names())
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

    scene2 = global_scene.resample(a)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
    img = get_enhanced_image(scene2['HRV']).data.transpose('y', 'x', 'bands')
    ax.imshow(img, transform=crs, extent=crs.bounds, origin='upper', cmap='jet')
    if coastlines:
        ax.coastlines()

    if save:
        if save_name is None:
            save_name = f'{filename[-32:-28]}-{filename[-28:-26]}-{filename[-26:-24]}_{filename[-24:-22]}_{filename[-22:-20]}'
            if coastlines:
                save_name = 'coastlines_' + save_name
        plt.savefig('images/' + save_name + '.png')


produce_image(
    'data/MSG3-SEVI-MSG15-0100-NA-20150414124241.311000000Z-NA/MSG3-SEVI-MSG15-0100-NA-20150414124241.311000000Z-NA.nat'
    , coastlines=True)
