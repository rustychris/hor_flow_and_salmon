"""
Develop a streamline-based interpolation method
"""
import matplotlib.pyplot as plt
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
clip=(646966, 647602, 4185504, 4186080)

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

g.plot_edges(clip=clip,ax=ax,color='k',lw=0.4)

cell_mask=g.cell_clip_mask(clip)

cc=g.cells_centroid()
u=ds.ucxa.values
v=ds.ucya.values

ax.quiver( cc[cell_mask,0],cc[cell_mask,1],
           u[cell_mask],v[cell_mask] )

##

# Trace a streamline for a single sample:
xy0=xyz_input[0,:2]

ax.plot([xy0[0]], [xy0[1]],'go',label='Start')

##

ds=xr.open_dataset('merged_map.nc')
g=unstructured_grid.UnstructuredGrid.from_ugrid(ds)

##

# Convert a mixed quad grid to all triangles
gtri=g.copy()
gtri.make_triangular()

mixed_to_tri=np.zeros( gtri.Ncells(), np.int32)
# Use centroid, since quads will have yield some bad circumcenters for triangles.
cent_tri=gtri.cells_centroid()

for c in range(gtri.Ncells()):
    if c%1000==0:
        print("%d/%d"%(c,gtri.Ncells()))
    mixed_to_tri[c] = g.select_cells_nearest(cent_tri[c],inside=True)


##

from stompy.model.pypart import basic

class Streamlines(basic.UgridParticles):
    # normalize all velocities to unit speed
    unit_speed=True
    # install negated velocities for back tracking
    reverse=False

    def __init__(self,nc,**kw):
        """
        nc: xr dataset, with 2D velocity in cell_{east,north}_velocity.
        """
        times=[np.datetime64("0000-01-01 00:00:00"),
               np.datetime64("4000-01-01 00:00:00")]

        nc_time=nc.copy()

        for v in ['cell_east_velocity','cell_north_velocity']:
            nc_time[v]=xr.concat([nc[v],nc[v]], dim='time' )
        nc_time['time']=('time',),times
        super(Streamlines,self).__init__([nc_time],**kw)

    def update_particle_velocity_for_new_step(self):
        # face, layer, time.
        # assumes 2D here.
        u=self.current_nc.cell_east_velocity.values[self.nc_time_i,:]
        v=self.current_nc.cell_north_velocity.values[self.nc_time_i,:]

        self.U=np.array( [u,v] ).T # again assume 2D

        if self.unit_speed:
            mag=utils.mag(self.U)
            self.U[mag>0] /= mag[mag>0][:,None]
        if self.reverse:
            self.U*=-1 # note this is not a view, so this is safe.

        # A little dicey - this overwrites any memory of convergent edges.
        # so every input interval, it's going to forget
        self.P['u']=self.U[ self.P['c'] ]


ds2=ds.rename({'ucxa':'cell_east_velocity',
               'ucya':'cell_north_velocity'})

##

ptm_fwd=Streamlines(ds2,grid=g)
ptm_rev=Streamlines(ds2,grid=g,reverse=True)

##

# debugging a crash
parts=xyz_input[::30,:2][518:519]
ptm=Streamlines(ds2,grid=g,record_dense=True)
ptm.add_particles(x=parts)
t0=np.datetime64("2000-01-01 00:00")
t_out=utils.to_unix(t0+np.timedelta64(10,'s')*np.arange(31))
ptm.set_time(t_out[0])
ptm.integrate(t_out)

# 946685100.0
locs=np.array( [coord for coord,time in ptm.dense] )

##

plt.figure(1).clf()
ax=plt.gca()
ptm.g.plot_edges(clip=clip,color='k',lw=0.4,ax=ax)

ax.plot(locs[:,0,0],locs[:,0,1],'g-o')

##
ptm.integrate([946685100.0 + 20.0])

##
parts=xyz_input[::30,:2]

for ptm in [ptm_fwd,ptm_rev]:
    ptm.add_particles(x=parts)

    t0=np.datetime64("2000-01-01 00:00")
    t_out=utils.to_unix(t0+np.timedelta64(10,'s')*np.arange(400))
    ptm.set_time(t_out[0])
    ptm.integrate(t_out)

##

locs_fwd=np.array( [coord for coord,time in ptm_fwd.output] )
locs_rev=np.array( [coord for coord,time in ptm_rev.output] )

locs=np.concatenate( (locs_fwd[::-1],locs_rev[1:]), axis=0)

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

cell_mask=g.cell_clip_mask(clip)

cc=g.cells_center()
u=ds.ucxa.values
v=ds.ucya.values
ax.quiver( cc[cell_mask,0],cc[cell_mask,1],
           u[cell_mask],v[cell_mask] )

from matplotlib import collections

segs=[]

for p in range(len(ptm.P)):
    seg=locs[:,p,:]
    sel=utils.within_2d(seg,clip)
    segs.append(seg[sel])

lcoll=collections.LineCollection(segs,color='b',lw=0.4)
ax.add_collection(lcoll)
ax.axis('equal')

# HERE - can get some tracks for many samples, but some fail.
#  may need to diffuse out the velocity field, for cases where
#  a sample is too close?
# this is failing on a particle 518, 
