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
gtri.modify_max_sides(3)

mixed_to_tri=np.zeros( gtri.Ncells(), np.int32)-1
# Use centroid, since quads will have yield some bad circumcenters for triangles.
cent_tri=gtri.cells_centroid()

for c in range(gtri.Ncells()):
    if c%1000==0:
        print("%d/%d"%(c,gtri.Ncells()))
    mixed_to_tri[c] = g.select_cells_nearest(cent_tri[c],inside=True)
assert np.all(mixed_to_tri>=0)

##

from stompy.model.pypart import ketefian

class StreamlinesKGS(ketefian.ParticlesKGS):
    def __init__(self,nc,**kw):
        """
        nc: xr dataset, with 2D velocity in cell_{east,north}_velocity.
        """
        times=[np.datetime64("0000-01-01 00:00:00"),
               np.datetime64("4000-01-01 00:00:00")]

        nc_time=nc.copy()

        # 3D
        for v in ['cell_east_velocity','cell_north_velocity']:
            new_var=xr.concat([nc[v],nc[v]], dim='time' )
            new_var=new_var.transpose('face','nMeshGlobal_layers','time')
            nc_time[v]=new_var
        # 2D
        for v in ['surface']:
            new_var=xr.concat([nc[v],nc[v]], dim='time' )
            new_var=new_var.transpose('face','time')
            nc_time[v]=new_var

        nc_time['time']=('time',),times
        super(StreamlinesKGS,self).__init__([nc_time],**kw)


##

ds2=ds.rename({'ucxa':'cell_east_velocity',
               'ucya':'cell_north_velocity'})

nc_tri=gtri.write_to_xarray()

# follow the order assumed in ketefian.py, face,layer, and time will be added in later.
# need edgedepth
new_u=np.zeros( (gtri.Ncells(),1), np.float64)
new_v=np.zeros( (gtri.Ncells(),1), np.float64)
new_u[:,0]=ds2['cell_east_velocity'].values[mixed_to_tri]
new_v[:,0]=ds2['cell_north_velocity'].values[mixed_to_tri]
new_depth=ds2['FlowElem_zcc'].values[mixed_to_tri]

# would be nice to pass edge normal velocities through,
# but have to get over DFM flow link vs net link
#new_unorm=np.zeros( (gtri.Nedges(),1), np.float64)
#new_unorm[:,0]

layer='nMeshGlobal_layers'
nc_tri['cell_east_velocity']=('face',layer), new_u
nc_tri['cell_north_velocity']=('face',layer), new_v
nc_tri['edge_normal_velocity']=('edge',layer),
nc_tri['depth']=('face',),new_depth
nc_tri['surface']=('face',),ds2['s1'].values[mixed_to_tri]
# kludge - hardwire the sign of edgedepth, positive down
nc_tri['edgedepth']=('edge',),-(ds2['NetNode_z'].values[ gtri.edges['nodes'] ].mean(axis=1))

##

ptm=StreamlinesKGS(nc_tri,grid=gtri)
# even that init is slow
# probably a lot of this would be faster by using DFM's unorm.

parts=xyz_input[::100,:2]
ptm.add_particles(x=parts)

t0=np.datetime64("2000-01-01 00:00")
t_out=utils.to_unix(t0+np.timedelta64(30,'s')*np.arange(5))
ptm.set_time(t_out[0])

##
ptm.integrate(t_out)


##

# 946685100.0
locs=np.array( [coord for coord,time in ptm.output] )

##

plt.figure(1).clf()
ax=plt.gca()
#ptm.g.plot_edges(clip=clip,color='k',lw=0.4,ax=ax)
# That looks pretty random.
ecoll=ptm.g.plot_edges(clip=clip,values=ptm.u_edge,lw=1.4,ax=ax)
plot_utils.cbar(ecoll)

ax.plot(locs[:,0,0],locs[:,0,1],'g-o')

##

parts=xyz_input[::30,:2]

##

zoom2=(647330.381025611, 647505.1716820023, 4185624.7873159577, 4185783.3777125655)

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

cell_mask=gtri.cell_clip_mask(zoom2)

cc=ptm.g.cells_center()
uv=ptm.U
ax.quiver( cc[cell_mask,0],cc[cell_mask,1],
           uv[cell_mask,0],uv[cell_mask,1] )

ax.plot(locs[:,0,0],locs[:,0,1],'g-o')

ax.axis('equal')

##
from matplotlib import collections

segs=[]

for p in range(len(ptm.P)):
    seg=locs[:,p,:]
    sel=utils.within_2d(seg,clip)
    segs.append(seg[sel])

lcoll=collections.LineCollection(segs,color='b',lw=0.4)
ax.add_collection(lcoll)
ax.axis('equal')

##

# 
