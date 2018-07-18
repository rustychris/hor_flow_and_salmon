"""
Develop a streamline-based interpolation method

This is taking a step back on the particle tracking to
just get some node velocities, and then do some
basic linear interpolation.  The point is that we just
need something close, not exact, but it's got to be reasonably
fast.

The interpolation method is based on radial basis functions in
the time,x,y space.
"""
import six
import scipy.spatial
from scipy.interpolate import Rbf
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

src='adcp' # 'dem'

if src=='adcp':
    # Rather than use the ADCP data directly, during testing
    # use its horizontal distribution, but pull "truth" from the
    # DEM
    xyz_dense=adcp_xyz.copy()
elif src=='dem':
    # Rather than use the ADCP data directly, during testing
    # use its horizontal distribution, but pull "truth" from the
    # DEM
    xyz_dense=adcp_xyz.copy()
    xyz_dense[:,2] = dem( xyz_dense[:,:2] )

##

# There are some repeated values in there, and the RBF is probably
# poorly conditioned even for closely spaced samples.
# round all coordinates to 10cm, and remove dupes.
# go even further, round to 1m but average.

xyz_quant=xyz_dense.copy()

xyz_quant[:,0] = np.round(xyz_quant[:,0],0)
xyz_quant[:,1] = np.round(xyz_dense[:,1],0)
counts=np.zeros(len(xyz_quant),np.int32)

rep={}
for i,xyz in enumerate(xyz_quant):
    k=tuple(xyz[:2])
    if k in rep:
        i=rep[k]
        xyz_quant[i,2]+=xyz[2]
    else:
        rep[k]=i
    counts[i]+=1

sel=counts>0
xyz_quant=xyz_quant[sel]
xyz_quant[:,2] /= counts[sel]

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
tile=dem.extract_tile(clip)

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
# bounds on the integration:
max_t=t0+150 # seconds
max_dist=60 # meters

tracks=[]

xyzs=xyz_quant[::1]

stream_dim='distance' # or 'time'

for i,y0 in enumerate(xyzs):
    if i%250==0:
        print("%d/%d"%(i,len(xyzs)))

    txys=[]
    for diff in [diff_fwd,diff_rev]:
        ivp=scipy.integrate.RK45(diff,t0=t0,y0=y0[:2],t_bound=max_t,max_step=10)

        d=0.0
        output=[]
        if stream_dim=='time':
            rec=lambda: output.append( (ivp.t, ivp.y[0], ivp.y[1]) )
        else:
            rec=lambda: output.append( (d, ivp.y[0], ivp.y[1]) )
        rec()
        while ivp.status=='running':
            ivp.step()
            rec()
            d+=utils.dist( output[-2][1:], output[-1][1:] )
            if d>=max_dist:
                break
        txys.append(np.array(output))
    # concatenate forward and backward
    fwd,rev=txys
    rev[:,0]*=-1 # negate time for reverse
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

ax.plot(xys[:,0],xys[:,1],'bo',ms=3)
lcoll=collections.LineCollection([track[:,1:] for track in tracks],
                                 color='b',lw=0.4)
ax.add_collection(lcoll)
ax.axis('equal')


## query

all_txy=np.concatenate(tracks)
if stream_dim=='time':
    all_txy[:,0] *= 0.1 # compress time - this will have to be tuned.
else: # stream_dim=='distance':
    # all_txy[:,0] *= (1./40) # 1:40 anisotropy
    all_txy[:,0] *= (1./10) # 1:10 anisotropy

# convert an index of all_xy back to a track and the index within that track
all_to_track_i=np.zeros( (len(all_txy),2), np.int32)
all_to_track_i[:,0]=np.concatenate( [i*np.ones(len(track)) for i,track in enumerate(tracks)])
all_to_track_i[:,1]=np.concatenate( [np.arange(len(track)) for i,track in enumerate(tracks)])

kdt_txy=scipy.spatial.KDTree(all_txy)

## query

# target=[0,647376,4.18578e6] # works great
# target=[0,647380, 4185780.0] # better
# target=[0,647409.8622228608, 4185783.646267513]
target=[0, 647218, 4185888]

nbr_dists,nbrs=kdt_txy.query(target,k=1000)
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
if 0: # Time/space at origins -- not sure why there is as much variation in time, though.
    scat=ax.scatter(nbr_origins[:,0],nbr_origins[:,1],20,nbr_t,cmap='seismic',lw=0.4)
if 0: # depth/space at origins
    scat=ax.scatter(nbr_origins[:,0],nbr_origins[:,1],20,nbr_z,cmap='jet',lw=0.4)
    scat.set_clim([-2,1])
if 1: # time/space at nbr
    scat=ax.scatter(all_txy[nbrs,1], all_txy[nbrs,2], 20, nbr_t,cmap='jet',lw=0.4)
if 0: # depth/space at nbr
    scat=ax.scatter(all_txy[nbrs,1], all_txy[nbrs,2], 20, nbr_z,cmap='jet',lw=0.4)
    scat.set_clim([-3,1])

scat.set_edgecolor('k')

# ax.plot(all_txy[nbrs,1],all_txy[nbrs,2],'bo',ms=10,alpha=0.4)

ax.axis('equal')

# smaller than default epsilon appears better
# 0.5 when there aren't too many samples, but 0.1 when it gets really
# dense.
rbf=Rbf(all_txy[nbrs,0],
        all_txy[nbrs,1],
        all_txy[nbrs,2],
        nbr_z,
        epsilon=0.5)

# fabricate t,x,y for output:
xs=np.linspace(target[1]-20,target[1]+20,60)
ys=np.linspace(target[2]-20,target[2]+20,60)
X,Y=np.meshgrid(xs,ys)
T=0*X

z_pred=rbf(T.ravel(),X.ravel(),Y.ravel())

del ax.collections[2:]
pm=ax.pcolormesh( X,Y,z_pred.reshape(X.shape),cmap='jet',zorder=-1)
pm.set_clim([-2,4])

ax.axis( (target[1]-30,target[1]+30,target[2]-30,target[2]+30) )

##
from sklearn import linear_model

# with 350 neighbors it's two minutes for about 180x180
# this was after quantizing inputs to 1m.
def recon(xy,k=1000,eps=0.5,interp='rbf'):
    target=[0,xy[0],xy[1]]
    nbr_dists,nbrs=kdt_txy.query(target,k=k)
    nbrs=np.array(nbrs)
    nbr_z=xyzs[all_to_track_i[nbrs,0],2]
    if interp=='rbf':
        try:
            rbf=Rbf(all_txy[nbrs,0],
                    all_txy[nbrs,1],
                    all_txy[nbrs,2],
                    nbr_z,
                    epsilon=eps)
        except np.linalg.LinAlgError:
            print("Linear algebra error")
            return np.nan

        z_pred=rbf(*target)
    elif interp=='mlr':
        clf=linear_model.LinearRegression()
        clf.fit(all_txy[nbrs,:], nbr_z)
        z_pred=clf.predict([target])[0]
    return z_pred

##

# try reconstructing a patch area:
if 0: # smaller patch
    rect=(647100, 647420, 4185680, 4186050)
    dx=2.0
    dy=2.0
    xs=np.linspace(rect[0],rect[1], 1+int(round((rect[1]-rect[0])/dx)))
    ys=np.linspace(rect[2],rect[3], 1+int(round((rect[3]-rect[2])/dy)))
else: # copy dem tile
    xs,ys=tile.xy()
    rect=[xs[0],xs[-1],ys[0],ys[-1]]

zs=np.zeros( (len(ys),len(xs)), np.float64 )

for row in range(len(ys)):
    print("%d/%d rows"%(row,len(ys)))
    for col in range(len(xs)):
        xy=[xs[col],ys[row]]
        c=g.select_cells_nearest(xy,inside=True)
        if c is None:
            zs[row,col]=np.nan
        else:
            zs[row,col]=recon(xy,k=350)

##

plt.figure(4).clf()
fig,(ax,ax_dem)=plt.subplots(1,2,sharex=True,sharey=True,num=4)

g.plot_edges(clip=clip,ax=ax,color='k',lw=0.4)
g.plot_edges(clip=clip,ax=ax_dem,color='k',lw=0.4)

#f=field.SimpleGrid(extents=rect,F=zs)
#f.plot(ax=ax,vmin=-7,vmax=2,cmap='jet')
fmed.plot(ax=ax,vmin=-7,vmax=2,cmap='jet')

tile.plot(ax=ax_dem,vmin=-7,vmax=2,cmap='jet')

ax.plot(xyz_quant[:,0],xyz_quant[:,1],'k.',ms=1.)

ax.axis('equal')

##
# f.write_gdal('tile-mlr.tif')
# f.write_gdal('tile-rbf.tif')
f.write_gdal('adcp-rbf.tif')

##

# has a fair bit of noise - run a 3x3 median filter:
from scipy import ndimage

Fmed=ndimage.median_filter(f.F,size=3)
fmed=field.SimpleGrid(extents=f.extents,F=Fmed)

fmed.write_gdal('adcp-rbf-med3.tif')


