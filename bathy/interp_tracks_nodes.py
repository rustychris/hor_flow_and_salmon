"""
Develop a streamline-based interpolation method

This is taking a step back on the particle tracking to
just get some node velocities, and then do some
basic linear interpolation.  The point is that we just
need something close, not exact, but it's got to be reasonably
fast.

"""
import six
import matplotlib.pyplot as plt
from matplotlib import collections
import scipy.integrate
import numpy as np
import xarray as xr
from stompy import utils
from stompy.grid import unstructured_grid
from stompy.model.delft import dfm_grid
from stompy.spatial import wkb2shp, proj_utils
import stompy.plot.cmap as scmap
cmap=scmap.load_gradient('hot_desaturated.cpt')
six.moves.reload_module(unstructured_grid)

##

import bathy
dem=bathy.dem()

##

adcp_shp=wkb2shp.shp2geom('derived/samples-depth.shp')
adcp_ll=np.array( [ np.array(pnt) for pnt in adcp_shp['geom']] )
adcp_xy=proj_utils.mapper('WGS84','EPSG:26910')(adcp_ll)
adcp_xyz=np.c_[ adcp_xy,adcp_shp['depth'] ]

##

# Rather than use the ADCP data directly, during testing
# use its horizontal distribution, but pull "truth" from the
# DEM
xyz_input=adcp_xyz.copy()
xyz_input[:,2] = dem( xyz_input[:,:2] )

##

ds=xr.open_dataset('merged_map.nc')
g=unstructured_grid.UnstructuredGrid.from_ugrid(ds)

##
# node velocity from average of the cells
node_U=np.zeros( (g.Nnodes(),2), np.float64 )

node_U[:,0]=g.interp_cell_to_node(ds.ucxa.values)
node_U[:,1]=g.interp_cell_to_node(ds.ucya.values)

# set boundary cells to 0, rough attempt to keep particles
# in the domain.
e2c=g.edge_to_cells(recalc=True)

boundary_edge=np.any(e2c<0,axis=1)
boundary_node=np.unique(g.edges['nodes'][boundary_edge])
node_U[boundary_node,:]=0.0

##

clip=(646966, 647602, 4185504, 4186080)

##

tri=g.mpl_triangulation()
finder=tri.get_trifinder()
u_coeffs=tri.calculate_plane_coefficients(node_U[:,0])
v_coeffs=tri.calculate_plane_coefficients(node_U[:,1])

##

def diff_fwd(time,x):
    """ velocity at location x[0],x[1] """
    t=finder(x[0],x[1])
    u=u_coeffs[t,0]*x[0]+u_coeffs[t,1]*x[1]+u_coeffs[t,2]
    v=v_coeffs[t,0]*x[0]+v_coeffs[t,1]*x[1]+v_coeffs[t,2]
    return [u,v]

def diff_rev(time,x):
    """ reverse velocities """
    t=finder(x[0],x[1])
    u=u_coeffs[t,0]*x[0]+u_coeffs[t,1]*x[1]+u_coeffs[t,2]
    v=v_coeffs[t,0]*x[0]+v_coeffs[t,1]*x[1]+v_coeffs[t,2]
    return [-u,-v]

##

# this is reasonably fast - probably can do the full dataset in a minute?
t0=0
max_t=t0+150 # seconds
max_dist=60 # meters

tracks=[]

xyzs=xyz_input[::1]

for i,y0 in enumerate(xyzs):
    if i%250==0:
        print("%d/%d"%(i,len(xyzs)))

    txys=[]
    for diff in [diff_fwd,diff_rev]:
        ivp=scipy.integrate.RK45(diff,t0=t0,y0=y0[:2],t_bound=max_t,max_step=10)

        d=0.0
        output=[]
        rec=lambda: output.append( (ivp.t, ivp.y[0], ivp.y[1]) )
        rec()
        while ivp.status=='running':
            ivp.step()
            rec()
            d+=utils.dist( output[-2][1:], output[-1][1:] )
            if d>=max_dist:
                # print(".",end="",flush=True)
                break
        #if d<max_dist:
        #    print("-",end="",flush=True)
        txys.append(np.array(output))
    # concatenate forward and backward
    fwd,rev=txys
    rev[:,0]*=-1 # negative time for reverse
    rev=rev[::-1]
    track=np.concatenate( (rev[:-1], fwd[:]),axis=0 )
    tracks.append(track)

##

cc=g.cells_centroid()
u=ds.ucxa.values
v=ds.ucya.values

##
plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

g.plot_edges(clip=clip,ax=ax,color='k',lw=0.4)

cell_mask=g.cell_clip_mask(clip)

#ax.quiver( cc[cell_mask,0],cc[cell_mask,1],
#           u[cell_mask],v[cell_mask] )

# ax.plot(output[:,1],output[:,2],'g-o')

ax.plot(xys[:,0],xys[:,1],'bo',ms=3)
lcoll=collections.LineCollection([track[:,1:] for track in tracks],
                                 color='b',lw=0.4)
ax.add_collection(lcoll)
ax.axis('equal')

##

# Construct the set of relevant points for a single target point:
import scipy.spatial

# Can I make an index with all of the vertices?
all_xy=np.concatenate(tracks)[:,1:]
# convert an index of all_xy back to a track and the index within that track
all_to_track_i=np.zeros( (len(all_xy),2), np.int32)
all_to_track_i[:,0]=np.concatenate( [i*np.ones(len(track)) for i,track in enumerate(tracks)])
all_to_track_i[:,1]=np.concatenate( [np.arange(len(track)) for i,track in enumerate(tracks)])
kdt=scipy.spatial.KDTree(all_xy)

## query

target=[647376,4.18578e6]
nbrs=kdt.query_ball_point(target,r=30)
nbrs=np.array(nbrs)

sel_tracks=np.unique( all_to_track_i[nbrs,0] )

nbr_dists=utils.dist( all_xy[nbrs],target )

# the closest node within each of the tracks
idx_in_track=np.zeros(len(sel_tracks),np.int32)-1

order=np.argsort(nbr_dists)
# for each track, I want the closest point along that track
close_nbrs=[] # a list of indices into all_xy corresponding to the closest neighbor per track.

for nbr in nbrs[order]:
    nbr_trk,nbr_idx=all_to_track_i[nbr]
    # map track index back to the index within just the selected tracks
    nbr_trk_i=np.searchsorted(sel_tracks,nbr_trk)
    if idx_in_track[nbr_trk_i]<0:
        idx_in_track[nbr_trk_i]=nbr_idx
        close_nbrs.append(nbr)
##

plt.figure(2).clf()
fig,ax=plt.subplots(num=2)
g.plot_edges(clip=clip,ax=ax,color='k',lw=0.4)
cell_mask=g.cell_clip_mask(clip)

ax.plot( [target[0]],[target[1]],'ro',zorder=4)

lcoll=collections.LineCollection([tracks[i][:,1:] for i in sel_tracks],
                                 color='b',lw=0.4)
ax.add_collection(lcoll)
ax.plot(all_xy[close_nbrs,0], all_xy[close_nbrs,1], 'go')

ax.axis('equal')

# HERE - this is fine for identifying the tracks, but still need to
# construct a flow-perpendicular line from the target, and intersect 
# each track with that segment.  put on hold until trying the 2D+T approach

##

# What about jumping to 3D?

# first, leave time as time, but scale it.

# Can I make an index with all of the vertices?
all_txy=np.concatenate(tracks)
all_txy[:,0] *= 0.1 # compress time

# convert an index of all_xy back to a track and the index within that track
all_to_track_i=np.zeros( (len(all_xy),2), np.int32)
all_to_track_i[:,0]=np.concatenate( [i*np.ones(len(track)) for i,track in enumerate(tracks)])
all_to_track_i[:,1]=np.concatenate( [np.arange(len(track)) for i,track in enumerate(tracks)])

kdt_txy=scipy.spatial.KDTree(all_txy)

## query

target=[0,647376,4.18578e6]
nbrs=kdt_txy.query_ball_point(target,r=10)
nbrs=np.array(nbrs)

nbr_origins=xyzs[all_to_track_i[nbrs,0],:2]
nbr_t=all_txy[nbrs,0]
nbr_z=xyzs[all_to_track_i[nbrs,0],2]

##

plt.figure(3).clf()
fig,ax=plt.subplots(num=3)
g.plot_edges(clip=clip,ax=ax,color='k',lw=0.4)
ax.plot( [target[1]],[target[2]],'ro',zorder=4)

# This is kind of a circle, but not exactly.
# ax.plot(all_txy[nbrs,1], all_txy[nbrs,2],'g.')
# Time/space -- not sure why there is as much variation in time, though.
# scat=ax.scatter(nbr_origins[:,0],nbr_origins[:,1],20,nbr_t,cmap='jet')
# depth/space
scat=ax.scatter(nbr_origins[:,0],nbr_origins[:,1],20,nbr_z,cmap='jet')

ax.axis('equal')

##

from scipy.interpolate import Rbf

rbf=Rbf(all_txy[nbrs,0],
        all_txy[nbrs,1],
        all_txy[nbrs,2],
        nbr_z )

# fabricate t,x,y for output:
xs=np.linspace(target[1]-10,target[1]+10,60)
ys=np.linspace(target[2]-10,target[2]+10,60)
X,Y=np.meshgrid(xs,ys)
T=0*X

z_pred=rbf(T.ravel(),X.ravel(),Y.ravel())

del ax.collections[2:]
pm=ax.pcolormesh( X,Y,z_pred.reshape(X.shape),cmap='jet',zorder=-1)
plt.setp([scat,pm],clim=[-2,1])


