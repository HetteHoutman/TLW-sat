import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from iris.coords import AuxCoord
from iris.experimental.stratify import relevel
import iris.plot as iplt
from prepare_data import read_variable
from prepare_metadata import get_sat_map_bltr
from cube_processing import add_pressure_to_cube

# w = read_variable('/storage/silver/metstudent/phd/sw825517/ukv_data/ukv_2023-04-19_12_000.pp', 150, 12)
# p = read_variable('/storage/silver/metstudent/phd/sw825517/ukv_data/ukv_2023-04-19_12_000.pp', 407, 12)
#
# add_pressure_to_cube(w, AuxCoord(points=p.data, standard_name=p.standard_name, long_name=p.long_name,
#                             units=p.units, coord_system=p.coord_system()))
#
# w = relevel(w, w.coords('air_pressure')[0], [80000])[0]

cube = read_variable('/storage/silver/metstudent/phd/sw825517/ukv_data/ukv_2023-04-19_12_000.pp', 33, 12)
regions = ['ireland', 'scotland', 'north_england', 'wales', 'england', 'cornwall']
letters = ['a', 'b', 'c', 'd', 'e', 'f']

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(7, 7))
iplt.pcolormesh(cube, cmap='Reds')
ax = plt.gca()
ax.coastlines()
ax.gridlines(draw_labels=True)

for region, letter in zip(regions, letters):
    sat_bl, sat_tr, _, _ = get_sat_map_bltr(region)
    rect = mpatches.Rectangle(sat_bl, sat_tr[0] - sat_bl[0], sat_tr[1] - sat_bl[1], edgecolor='k', facecolor='none', linewidth=2)
    ax.add_patch(rect)
    plt.annotate(letter.capitalize(), sat_bl, xytext=(0.25,  0.25), textcoords='offset fontsize', backgroundcolor='w')

plt.colorbar(location='bottom', label='Model orography (m)', pad=0.07)
plt.savefig('plots/regions_map.png', dpi=300, bbox_inches='tight')
plt.show()
