from pyresample.geometry import AreaDefinition
area_id = 'custom'
x_size = 11136
y_size = 11136
area_extent = (-5570248.686685662, -5567248.28340708, 5567248.2834070, 5570248.686685662)
projection = '+proj=geos +h=35785831.0'
description = "custom test"
proj_id = 'test'
areadef = AreaDefinition(area_id, description, proj_id, projection,x_size, y_size, area_extent)
lons, lats = areadef.get_lonlats()
print('wacht eens even')