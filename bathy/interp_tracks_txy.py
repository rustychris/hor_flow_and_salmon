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
from sklearn import linear_model
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

src='dem' # 'adcp' or 'dem'
cluster=True
quant_xy=False

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
from sklearn.cluster import AgglomerativeClustering

xyzs=xyz_dense.copy()

if cluster:
    linkage='complete'
    n_clusters=3000
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters)
    clustering.fit(xyzs[:,:2])

    group_xyz=np.zeros( (n_clusters,3) )
    for grp,members in utils.enumerate_groups(clustering.labels_):
        group_xyz[grp] = xyzs[members].mean(axis=0)
    xyzs=group_xyz

if quant_xy:
    quant_scale=1.0 # m

    # There are some repeated values in there, and the RBF is probably
    # poorly conditioned even for closely spaced samples.
    # round all coordinates to 10cm, and remove dupes.
    # go even further, round to 1m but average.
    xyz_quant=xyzs.copy()

    xyz_quant[:,0] = quant_scale*np.round(xyz_quant[:,0]/quant_scale)
    xyz_quant[:,1] = quant_scale*np.round(xyz_dense[:,1]/quant_scale)
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

    xyzs=xyz_quant[::1]

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

stream_dim='distance' # or 'time'

# this is reasonably fast - probably can do the full dataset in a minute?
t0=0
# bounds on the integration:
max_t=t0+150 # seconds
max_dist=60 # meters

tracks=[]

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
            d+=utils.dist( output[-1][1:], ivp.y )
            rec()
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

if 1:
    plt.figure(2).clf()
    fig,ax=plt.subplots(num=2)
    fig.set_size_inches((12,10),forward=True)

    g.plot_edges(clip=clip,ax=ax,color='k',lw=0.4)

    cell_mask=g.cell_clip_mask(clip)

    ax.plot(xyzs[:,0],xyzs[:,1],'go',ms=3,zorder=2)
    lcoll=collections.LineCollection([track[:,1:] for track in tracks],
                                     color='b',lw=0.4)
    ax.add_collection(lcoll)
    ax.axis('equal')
    fig.tight_layout()
    # fig.savefig('grid-samples-streamlines.png')

## Quantize the tracks, and scale 3rd dimension

stream_factor=0.1 # controls the anisotropy of the search

quant_tracks=[]
track_quant_scale=1.0
for track in tracks:
    # resample track to quant_scale, taking into account stream_factor
    # the goal is to quantize the 3D dataset equally in x,y, and t/d.
    # make sure 0 is still in there
    neg_d=-np.arange(0.0,-track[0,0],track_quant_scale)[::-1]
    pos_d=np.arange(0.0,track[-1,0],track_quant_scale)[1:]
    tgt_d=np.r_[neg_d,pos_d]
    qtrack=np.zeros( (len(tgt_d),3), np.float64)
    qtrack[:,0]=tgt_d * stream_factor
    qtrack[:,1]=np.interp(tgt_d, track[:,0],track[:,1])
    qtrack[:,2]=np.interp(tgt_d, track[:,0],track[:,2])
    quant_tracks.append(qtrack)

##
all_txy=np.concatenate(quant_tracks)

# convert an index of all_xy back to a track and the index within that track
all_to_track_i=np.zeros( (len(all_txy),2), np.int32)
all_to_track_i[:,0]=np.concatenate( [i*np.ones(len(track)) for i,track in enumerate(quant_tracks)])
all_to_track_i[:,1]=np.concatenate( [np.arange(len(track)) for i,track in enumerate(quant_tracks)])

kdt_txy=scipy.spatial.KDTree(all_txy)

## query

# target=[0,647376,4.18578e6] # worked great
target=[0,647380, 4185780.0] # better
# target=[0,647409.8622228608, 4185783.646267513]
# target=[0, 647218, 4185888]
# target=[0, 647164., 4185844.] # tricky spot with ADCP data - tip of barrier
# target=[0, 647159., 4185844.]

# Is it any better to choose only the "closest" of the samples along a track?
# otherwise some methods are blind to those samples being correlated.
# this removes a lot of spatial information, though.

nbr_dists,nbrs=kdt_txy.query(target,k=1000)
nbrs=np.array(nbrs)
if 0: # trim to a single, closest sample per track
    nbr_tracks=all_to_track_i[nbrs,0]
    slim_nbrs=[]
    for k,idxs in utils.enumerate_groups(nbr_tracks):
        best=np.argmin( nbr_dists[idxs] )
        slim_nbrs.append( nbrs[idxs[best]] )
    nbrs=np.array(slim_nbrs)

nbr_origins=xyzs[all_to_track_i[nbrs,0],:2]
nbr_t=all_txy[nbrs,0]
nbr_z=xyzs[all_to_track_i[nbrs,0],2]

# Look a bit more closely at the 3D distribution of those samples:
plt.figure(5).clf()
fig,axs=plt.subplots(2,3,num=5,sharex=True,sharey=True)

scats=[]
axs[0,0].axis('equal')

scats.append( axs[0,0].scatter(all_txy[nbrs,1],all_txy[nbrs,2],30,nbr_z) )
axs[0,0].set_xlabel('nbr-x') ; axs[0,0].set_ylabel('nbr-y')
axs[0,0].plot( [target[1]],[target[2]],'mo',ms=10)

axs[0,1]=fig.add_subplot(2,3,2)
scats.append( axs[0,1].scatter(all_txy[nbrs,1],nbr_t,30,nbr_z) )
axs[0,1].set_xlabel('nbr-x') ; axs[0,1].set_ylabel('nbr-t')
axs[0,1].plot( [target[1]],[target[0]],'mo',ms=10)

scats.append( axs[1,0].scatter(nbr_origins[:,0],nbr_origins[:,1],30,nbr_z) )
axs[1,0].set_xlabel('org-x') ; axs[1,0].set_ylabel('org-y')
axs[1,0].plot( [target[1]],[target[2]],'mo',ms=10)

img=tile.plot(ax=axs[1,1],cmap=cmap)
axs[1,1].plot( [target[1]],[target[2]],'mo',ms=10)
axs[1,1].set_title('DEM tile')

z_clim=[-6,3]

plt.setp(scats,cmap=cmap,clim=z_clim)
img.set_clim(z_clim)

# smaller than default epsilon appears better
# 0.5 when there aren't too many samples, but 0.1 when it gets really
# dense.
# smoothing doesn't do much
# linear slightly better behaved? thin_plate?
# maybe the time axis should be sampled at a compatible resolution to space?
# that seems to make everything better behaved, might introduce too much
# smoothing?
ax=axs[0,2]

# Show the local interpolated field
xs=np.linspace(target[1]-20,target[1]+20,60)
ys=np.linspace(target[2]-20,target[2]+20,60)
X,Y=np.meshgrid(xs,ys)
T=0*X
txy_pred=np.c_[T.ravel(),X.ravel(),Y.ravel()]

interp='rbf'
if interp=='rbf':
    t_fac=1 # can further change the anisotropy on top of the previous factor
    rbf=Rbf(t_fac*all_txy[nbrs,0],
            all_txy[nbrs,1],
            all_txy[nbrs,2],
            nbr_z, function='linear',
            epsilon=0.5)

    z_pred=rbf(T.ravel(),X.ravel(),Y.ravel())
    z_pred=z_pred.reshape(X.shape)
elif interp=='griddata':
    z_pred=scipy.interpolate.griddata(all_txy[nbrs,:],
                                      nbr_z,
                                      txy_pred,
                                      rescale=True)
    z_pred=z_pred.reshape(X.shape)
elif interp=='mlr':
    clf=linear_model.LinearRegression()
    clf.fit(all_txy[nbrs,:], nbr_z)
    z_pred=clf.predict(txy_pred)
    z_pred=z_pred.reshape(X.shape)
elif interp=='wmlr':
    clf=linear_model.LinearRegression()
    dists=utils.dist(all_txy[nbrs,:],target)
    weights=(dists+1.0)**-2
    clf.fit(all_txy[nbrs,:], nbr_z,weights)
    z_pred=clf.predict(txy_pred)
    z_pred=z_pred.reshape(X.shape)
elif interp=='idw': # bad.
    N=len(txy_pred)
    z_pred=np.zeros(N,np.float64)
    for i in range(N):
        delta=all_txy[nbrs,:]-txy_pred[i]
        delta[:,0] *= 10 # rescale streamdistance
        dists=utils.mag(delta)
        weights=(dists+0.1)**(-2)
        weights = weights / weights.sum()
        z_pred[i]=(weights*nbr_z).sum()
    z_pred=z_pred.reshape(X.shape)
elif interp=='krige':
    import sklearn.gaussian_process as GP
    points=all_txy[nbrs,:]
    values=nbr_z
    origin=points.mean(axis=0)
    # gp = GP.GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1., nugget=0.001)
    gp = GP.GaussianProcessRegressor(kernel=GP.kernels.ConstantKernel(1.0),
                                     n_restarts_optimizer=9)
    gp.fit(points, values)
    z_pred = gp.predict(txy_pred).reshape(X.shape)


pm=ax.pcolormesh( X,Y,z_pred,cmap=cmap,zorder=-1)
pm.set_clim(z_clim)
ax.plot( [target[1]],[target[2]],'mo',zorder=3)
axs[0,0].axis( (target[1]-30,target[1]+30,target[2]-30,target[2]+30) )
fig.tight_layout()

##

# with 350 neighbors it's two minutes for about 180x180
# this was after quantizing inputs to 1m.
def recon(xy,k=1000,eps=0.5,interp='rbf'):
    target=[0,xy[0],xy[1]]
    nbr_dists,nbrs=kdt_txy.query(target,k=k)
    nbrs=np.array(nbrs)
    if 0: # trim to a single, closest sample per track
        nbr_tracks=all_to_track_i[nbrs,0]
        slim_nbrs=[]
        for k,idxs in utils.enumerate_groups(nbr_tracks):
            best=np.argmin( nbr_dists[idxs] )
            slim_nbrs.append( nbrs[idxs[best]] )
        nbrs=np.array(slim_nbrs)

    nbr_z=xyzs[all_to_track_i[nbrs,0],2]
    if interp=='rbf':
        try:
            rbf=Rbf(all_txy[nbrs,0],
                    all_txy[nbrs,1],
                    all_txy[nbrs,2],
                    nbr_z,
                    epsilon=eps, function='linear')
        except np.linalg.LinAlgError:
            print("Linear algebra error")
            return np.nan

        z_pred=rbf(*target)
    elif interp=='mlr':
        clf=linear_model.LinearRegression()
        clf.fit(all_txy[nbrs,:], nbr_z)
        z_pred=clf.predict([target])[0]
    elif interp=='griddata':
        z_pred=scipy.interpolate.griddata(all_txy[nbrs,:],
                                          nbr_z,
                                          target,
                                          rescale=True)
    elif interp=='krige':
        points=all_txy[nbrs,:]
        values=nbr_z
        gp = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1., nugget=0.001)
        gp.fit(all_txy[nbrs,:],nbr_z)
        z_pred = gp.predict([target])[0]
    elif interp=='idw':
        delta=all_txy[nbrs,:]-np.asarray(target)
        delta[:,0] *= 10 # rescale streamdistance?
        dists=utils.mag(delta)
        weights=(dists+eps)**(-2)
        weights = weights / weights.sum()
        z_pred=(weights*nbr_z).sum()

    return z_pred

##

# try reconstructing a patch area:
if 0: # smaller patch
    rect=(647120, 647240, 4185800, 4185900)
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
            zs[row,col]=recon(xy,k=350,interp='rbf')

##

# griddata is not much of an improvement
#  - still has the quantized streamline noise.
# mlr still too smooth, too much streamline

plt.figure(4).clf()
fig,(ax,ax_dem)=plt.subplots(1,2,sharex=True,sharey=True,num=4)

g.plot_edges(clip=clip,ax=ax,color='k',lw=0.4)
g.plot_edges(clip=clip,ax=ax_dem,color='k',lw=0.4)

f=field.SimpleGrid(extents=rect,F=zs)
f.plot(ax=ax,vmin=-7,vmax=2,cmap='jet')
# fmed.plot(ax=ax,vmin=-7,vmax=2,cmap='jet')

tile.plot(ax=ax_dem,vmin=-7,vmax=2,cmap='jet')

ax.plot(xyz_quant[:,0],xyz_quant[:,1],'k.',ms=1.)

ax.axis('equal')
ax.axis(rect)

##
# f.write_gdal('tile-mlr.tif')
# f.write_gdal('tile-rbf.tif')
# f.write_gdal('adcp-rbf.tif')
# f.write_gdal('tile-rbf-qtracks.tif')
# f.write_gdal('adcp-rbf-qtracks.tif')
f.write_gdal('tile-rbf-cluster.tif')

##

# has a fair bit of noise - run a 3x3 median filter:
from scipy import ndimage

Fmed=ndimage.median_filter(f.F,size=3)
fmed=field.SimpleGrid(extents=f.extents,F=Fmed)

fmed.write_gdal('tile-rbf-cluster-med3.tif')


