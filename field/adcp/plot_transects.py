# Read the collection of csv files associated with a sontek transect
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import read_sontek

##

six.moves.reload_module(read_sontek)

rivr_fn='040518_7_BTref/20180405125420r.rivr'

ds=read_sontek.surveyor_to_xr(rivr_fn,proj='EPSG:26910')

##

# Transect of speed
plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
x,z,speed = xr.broadcast(ds.track_dist,-ds.location,ds.water_speed)
scat=ax.scatter(x, z, 40, speed, cmap='jet')
plt.colorbar(scat)

##

# Plan view, scatter and quiver
plt.figure(2).clf()
fig,ax=plt.subplots(num=2)
scat=ax.scatter(ds.x_utm, ds.y_utm, 40, ds.mean_water_speed, cmap='jet')

avg_east=ds.Ve.mean(dim='cell')
avg_north=ds.Vn.mean(dim='cell')
quiv=ax.quiver(ds.x_utm.values, ds.y_utm.values, avg_east.values, avg_north.values)

plt.colorbar(scat,label='Speed m/s')
ax.axis('equal')
