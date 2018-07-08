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
ds=all_ds[1]

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

coll=xr_transect.plot_scalar(ds,ds.Ve,ax=ax)

ax.plot(ds.d_sample,ds.z_bed,'k-')

##

@memoize()
def bathy():
    from stompy.spatial import field
    return field.GdalGrid('../bathy/OldRvr_at_SanJoaquinRvr2012_0104-utm-m_NAVD.tif')

##

# Transect averaging:

def transects_to_segment(trans,unweight=True,ax=None):
    """
    trans: list of transects per xr_transect
    unweight: if True, follow ADCPy and thin dense clumps of pointer.

    return a segment [ [x0,y0],[x1,y1] ] approximating the
    points

    if ax is supplied, it is a matplotlib Axes into which the steps
    of this method are plotted.
    """
    from stompy.spatial import linestring_utils
    all_xy=[]
    all_dx=[]

    all_dx=[get_dx_sample(tran).values
            for tran in trans]
    median_dx=np.median(np.concatenate(all_dx))

    for tran_i,tran in enumerate(trans):
        xy=np.c_[ tran.x_sample.values,tran.y_sample.values]

        if ax:
            ax.plot(xy[:,0],xy[:,1],marker='.',label='Input %d'%tran_i)

        if unweight:
            # resample_linearring() allows for adding new points, which can be
            # problematic if the GPS jumps around, adding many new points on a
            # rogue line segment.
            # downsample makes sure that clusters are thinned, but does not add
            # new points between large jumps.
            xy=linestring_utils.downsample_linearring(xy,density=3*median_dx,
                                                      closed_ring=False)
        all_xy.append(xy)

    all_xy=np.concatenate(all_xy)
    if ax:
        ax.plot(all_xy[:,0],all_xy[:,1],'bo',label='Unweighted')

    C=np.cov(all_xy.T)
    vec=utils.to_unit(C[:,0])

    centroid=all_xy.mean(axis=0)
    dist_along=np.dot((all_xy-centroid),vec)
    dist_range=np.array( [dist_along.min(), dist_along.max()] )

    seg=centroid + dist_range[:,None]*vec
    if ax:
        ax.plot(seg[:,0],seg[:,1],'k-o',lw=5,alpha=0.5,label='Segment')
        ax.legend()
    return seg

trans=all_ds

seg=transects_to_segment(all_ds)

##
dx=None
dz=None

#
if dx is None:
    # Define the target vertical and horizontal bins
    all_dx=[get_dx_sample(tran).values
            for tran in trans]
    median_dx=np.median(np.concatenate(all_dx))
    dx=median_dx

if dz is None:
    all_dz=[ np.abs(get_z_dz(tran).values.ravel())
             for tran in trans]
    all_dz=np.concatenate( all_dz )
    # generally want to retain most of the vertical
    # resolution, but not minimum dz since there could be
    # some partial layers, near-field layers, etc.
    # even 10th percentile may be small.
    dz=np.percentile(all_dz,10)

# Get the maximum range of valid vertical
z_bnds=[]
for tran in trans:
    V,z_full,z_dz = xr.broadcast(tran.Ve, tran.z_ctr, get_z_dz(tran))
    valid=np.isfinite(V.values)
    z_valid=z_full.values[valid]
    z_low=z_full.values[valid] - z_dz.values[valid]/2.0
    z_high=z_full.values[valid] + z_dz.values[valid]/2.0
    z_bnds.append( [z_low.min(), z_high.max()] )

z_bnds=np.concatenate(z_bnds)
z_min=z_bnds.min()
z_max=z_bnds.max()

# Resample each transect in the vertical:
new_z=np.linspace(z_min,z_max,int(round((z_max-z_min)/dz)))

##

six.moves.reload_module(xr_transect)
ds_resamp=[xr_transect.resample_z(tran,new_z)
           for tran in trans]

##

plt.figure(2).clf()
fig,axs=plt.subplots(2,1,num=2,sharex=True,sharey=True)

xr_transect.plot_scalar(all_ds[0],all_ds[0].Ve,ax=axs[0])
xr_transect.plot_scalar(ds_resamp[0],ds_resamp[0].Ve,ax=axs[1])

##

# Keep this general, so that segment is allowed to have more than
# just two vertices
new_xy = linestring_utils.resample_linearring(seg,dx)

##

# resampling a single transect onto the new horizontal coordinates.
# can only operate on transects which have a uniform z coordinate
ds_in=ds_resamp[0]
assert ds_in.z_ctr.ndim==1,"Resampling horizontal requires uniform vertical coordinate"

## 
new_ds=xr.Dataset()
new_ds['z'


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

