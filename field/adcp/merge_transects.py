"""
Average multiple transects.
"""

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import glob

import read_sontek
from stompy.memoize import memoize

import six
from stompy import xr_transect

six.moves.reload_module(xr_transect)
six.moves.reload_module(read_sontek)

#---

# Choose 2, because it has the matlab output which is richer and has
# RiverSurveyor computed flows.
adcp_data_dir="040518_BT/040518_2BTref"

rivr_fns=glob.glob('%s/*.rivr'%adcp_data_dir)

all_ds=[ read_sontek.surveyor_to_xr(fn,proj='EPSG:26910',positive='up')
         for fn in rivr_fns]


##
ds=all_ds[0]

fig=plt.figure(1)
fig.clf()

coll=xr_transect.plot_scalar(ds,ds.Ve)

# plt.axis(ymin=-7,ymax=0,xmin=0,xmax=100)

##

@memoize()
def bathy():
    from stompy.spatial import field
    return field.GdalGrid('../bathy/OldRvr_at_SanJoaquinRvr2012_0104-utm-m_NAVD.tif')

##

# x-z of a single transect:

ds=all_bt[0]

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

track_dist_3d,_ = xr.broadcast(ds.track_dist,ds.Ve)

track_z=-ds.location.values
scat=ax.scatter(track_dist_3d.values,track_z,
                40,ds.Ve.values)


##

x_3d,y_3d,_ = xr.broadcast(ds.x_utm,ds.y_utm,ds.Ve)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(2)
fig.clf()
ax = fig.add_subplot(111, projection='3d')


scat=ax.scatter(x_3d.values.ravel(),y_3d.values.ravel(),track_z.ravel(),
                s=40,c=ds.Ve.values.ravel())

##

all_xyzen=[]

for ds in all_bt:
    x_3d,y_3d,_ = xr.broadcast(ds.x_utm,ds.y_utm,ds.Ve)
    track_z=-ds.location.values

    xyzen = np.c_[x_3d.values.ravel(),
                  y_3d.values.ravel(),
                  track_z.ravel(),
                  ds.Ve.values.ravel(),
                  ds.Vn.values.ravel()]
    all_xyzen.append( xyzen )

##

combined=np.concatenate(all_xyzen)

## 

from scipy.interpolate import Rbf

rbf_ve=Rbf( combined[:,0],
            combined[:,1],
            combined[:,2],
            combined[:,3] )
#rbf_vn=Rbf( combined[:,0],
#            combined[:,1],
#            combined[:,2],
#            combined[:,4] )

##

