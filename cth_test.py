# from satpy import Scene
# global_scene = Scene(reader='grib', filenames=[r'data/MSG3-SEVI-MSGCLTH-0100-0100-20230419120000.000000000Z-20230419121431-4854768.grb'])

import pygrib
grbs = pygrib.open(r'data/MSG3-SEVI-MSGCLTH-0100-0100-20230419120000.000000000Z-20230419121431-4854768.grb')
grb = grbs.read(1)[0]