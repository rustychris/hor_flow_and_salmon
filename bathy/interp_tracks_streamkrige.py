"""
Develop a streamline-based interpolation method
 -- take a look at kriging.  Is anything gained by
    advecting the sample point over the kriged field?
 -- then we accumulate data along the trajectory
    according to the uncertainty and the distance/time
    from the sample.
 -- will this still be able to extrapolate?  I think so,
    since the kriged surface can extrapolate.  both methods
    do very little in areas where there is no flow.
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
from stompy import utils, filters
from stompy.grid import unstructured_grid
from stompy.model.delft import dfm_grid
from stompy.spatial import wkb2shp, proj_utils, linestring_utils
import stompy.plot.cmap as scmap
cmap=scmap.load_gradient('hot_desaturated.cpt')
six.moves.reload_module(unstructured_grid)

##

import bathy
dem=bathy.dem()

##

if 0:
    adcp_shp=wkb2shp.shp2geom('derived/samples-depth.shp')
    adcp_ll=np.array( [ np.array(pnt) for pnt in adcp_shp['geom']] )
    adcp_xy=proj_utils.mapper('WGS84','EPSG:26910')(adcp_ll)
    adcp_xyz=np.c_[ adcp_xy,adcp_shp['depth'] ]
if 1: # fake, much sparser tracks.
    adcp_shp=wkb2shp.shp2geom('sparse_fake_bathy_trackline.shp')
    xys=[]
    for feat in adcp_shp['geom']:
        feat_xy=np.array(feat)
        feat_xy=linestring_utils.resample_linearring(feat_xy,1.0,closed_ring=0)
        feat_xy=filters.lowpass_fir(feat_xy,winsize=6,axis=0)
        xys.append(feat_xy)
    adcp_xy=np.concatenate(xys)
    adcp_xyz=np.c_[ adcp_xy, np.nan*adcp_xy[:,0]]
    
##     
src='dem' # 'adcp' or 'dem'
cluster=False
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

xyzs=xyz_dense.copy()

##

# Brief foray in to kriging of the original data

points=xyzs[:,:2]
values=xyzs[:,2]
origin=points.mean(axis=0)
points=points - origin

# Straight up this chews up GB of RAM and doesn't finish.
# 18233 points... 

## 
plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
ax.scatter(points[:,0],points[:,1],30,values)
ax.axis('equal')

# 900 points
# clip=(-122.74664058399362, -49.27590162282269, 68.94499228082248, 132.57052411179984)
# 250 points
# clip=(-124.24120185126887, -59.71677290460141, 94.1410073302137, 150.0190502881254)
# SJ downstream, a bit sparser
# clip=(9.959235332220862, 123.1602687552361, 164.5653277311473, 264.6270573251136)
# same but zoomed out for sparse data
# clip=(-50.46658407434541, 168.1397882947122, 105.21457983858255, 298.44719493115156)
# zoom way out: blows up
# clip=(-356.8142432110379, 289.9372478568944, -309.4875440254, 262.1951593639185)

# the junction and no more
# clip=(-193.2320131408508, 26.6068137657644, -30.4523597432862, 133.36302417744963)
# just upstream of junction
clip=(-119.87939887615079,223.2856714502928, -262.16947063461413, 69.84122995157253)

sel=utils.within_2d(points,clip)

# drop duplicates:
sel_points=points[sel]
for iidx,idx in enumerate(np.nonzero(sel)[0]):
    this_point=sel_points[iidx]
    rest_of_the_points=sel_points[iidx+1:]
    match_xy=np.all(rest_of_the_points==this_point,axis=1)
    if np.any(match_xy):
        print("Dupe %s"%str(this_point))
        sel[idx]=False

##

# can't deal with large number of points, say 2000.
import sklearn.gaussian_process as GP
from sklearn.gaussian_process import kernels

##

# Dangerous territory -- try a specialized RBF that allows rotating
# the covariance matrix

# essentially use the Mahalonobis distance.
# there are two length scales and an angle.
# same as the dofs in a symmetric 2x2 matrix
# can stick with the more geometric interpretation,
from scipy.spatial.distance import pdist, cdist, squareform

class RotRBF2D(kernels.StationaryKernelMixin, kernels.NormalizedKernelMixin, kernels.Kernel):
    """
    Rotated Radial-basis function kernel in 2 dimensions
    """
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 angle=0.0, angle_bounds=(-2*np.pi,2*np.pi)):
        # these angle bounds are very generous, as a kludge to avoid running
        # into hard boundaries.  probably a better way to do that.
        self.length_scale = length_scale
        assert len(self.length_scale)==2,"Only ready for 2D anisotropic"
        self.length_scale_bounds = length_scale_bounds
        self.angle=angle
        self.angle_bounds=angle_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return kernels.Hyperparameter("length_scale", "numeric",
                                          self.length_scale_bounds,
                                          len(self.length_scale))
        return kernels.Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_angle(self):
        return kernels.Hyperparameter("angle","angle", self.angle_bounds)
        
    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = kernels._check_length_scale(X, self.length_scale)
        # compute inverse covariance matrix with length_scale and
        # angle
        # d = sqrt( (u-v) IV (u-v).T )
        # so if there is no rotation, then IV just has the
        # 
        Lx,Ly=length_scale
        cs=np.cos(self.angle)
        sn=np.sin(self.angle)
        VI = np.array( [[cs/Lx**2,-sn/(Lx*Ly)],
                        [sn/(Lx*Ly),cs/Ly**2 ]] )
        
        if Y is None:
            dists = pdist(X, metric='mahalanobis', VI=VI)
            K = np.exp(-.5 * dists**2)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric='mahalanobis', VI=VI)
            K = np.exp(-.5 * dists**2)

        if eval_gradient:
            assert self.anisotropic
            assert not self.hyperparameter_length_scale.fixed
            assert not self.hyperparameter_angle.fixed
            
            # if self.hyperparameter_length_scale.fixed:
            #     # Hyperparameter l kept fixed
            #     return K, np.empty((X.shape[0], X.shape[0], 0))
            # elif not self.anisotropic or length_scale.shape[0] == 1:
            #     K_gradient = \
            #         (K * squareform(dists))[:, :, np.newaxis]
            #     return K, K_gradient
            # elif self.anisotropic:
            
            # approximate gradient numerically
            def f(theta):  # helper function
                return self.clone_with_theta(theta)(X, Y)
            return K, kernels._approx_fprime(self.theta, f, 1e-10)
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}],angle={2:.3g})".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format,self.length_scale)),
                self.angle)
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0])

    # Have to additionally supply a specialized theta getter/setter, as the
    # stock implementation log-transforms everything which is no good for
    # angle.
    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.
        """
        theta = []
        params = self.get_params()
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                if hyperparameter.value_type=='numeric':
                    theta.append(np.log(params[hyperparameter.name]))
                elif hyperparameter.value_type=='angle':
                    theta.append(params[hyperparameter.name])
                else:
                    raise Exception("Unknown value_type %s"%hyperparameter.value_type)
        if len(theta) > 0:
            return np.hstack(theta)
        else:
            return np.array([])

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        params = self.get_params()
        i = 0
        for hyperparameter in self.hyperparameters:
            if hyperparameter.fixed:
                continue
            if hyperparameter.n_elements > 1:
                # vector-valued parameter
                if hyperparameter.value_type=='numeric':
                    params[hyperparameter.name] = np.exp(
                        theta[i:i + hyperparameter.n_elements])
                elif hyperparamater.value_type=='angle':
                    params[hyperparameter.name] = theta[i:i + hyperparameter.n_elements]
                else:
                    raise Exception("Unknown value_type %s"%hyperparameter.value_type)
                i += hyperparameter.n_elements
            else:
                params[hyperparameter.name] = np.exp(theta[i])
                i += 1

        if i != len(theta):
            raise ValueError("theta has not the correct number of entries."
                             " Should be %d; given are %d"
                             % (i, len(theta)))
        self.set_params(**params)

    @property
    def bounds(self):
        """ just like theta, have to avoid log-transform on the rotation
        """
        bounds = []
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                if hyperparameter.value_type=='numeric':
                    bounds.append(np.log(hyperparameter.bounds))
                elif hyperparameter.value_type=='angle':
                    bounds.append(hyperparameter.bounds)
                else:
                    raise Exception("Unknown value_type %s"%hyperparameter.value_type)
        if len(bounds) > 0:
            return np.vstack(bounds)
        else:
            return np.array([])


## 
# This is obsolete, though it seems to work.
# gp = GP.GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1., nugget=0.001)
# kernel=GP.kernels.ConstantKernel(1.0) ==> fails
# alpha: just guessing that it is kind of like nugget.
# normalize_y: values are not already normalized and centered, so
#    ask them to do it.

from sklearn.gaussian_process.kernels import ConstantKernel,RBF
# This is the default -- fits only right around the points
#kernel=( ConstantKernel(constant_value=1.0,constant_value_bounds=(0.1,10.0))
#         * RBF(length_scale=1.0,length_scale_bounds=(0.1,10.)) )
# At least does something...
#kernel=( ConstantKernel(constant_value=10.0,constant_value_bounds=(0.1,10.0))
#         * RBF(length_scale=[100.0,100.0],length_scale_bounds=(1,1000.)) )
# getting crazy
kernel=( ConstantKernel(constant_value=10.0,constant_value_bounds=(0.1,10.0))
         * RotRBF2D(length_scale=[100.0,100.0],length_scale_bounds=(1,1000.)) )

##
for hyperparameter in kernel.hyperparameters: print(hyperparameter)

##

# STUCK HERE
#  My rotated kernel is not behaving, and in particular it is allowing
#  nan to creep into the optimization steps.  probably related to some
#  extra step where it is log-transforming and I don't special case it.

# seems that the new Gaussian Process is much harder to use for the
# basic kriging method.
# alpha is very important -- otherwise a bit of noise along track
# gets fed into a very rigid system
# how about two length scales?? not much better
gp = GP.GaussianProcessRegressor(kernel=kernel, alpha=1.0,
                                 normalize_y=True,
                                 n_restarts_optimizer=9)
ret=gp.fit(points[sel], values[sel])

xs=np.arange(clip[0],clip[1],1)
ys=np.arange(clip[2],clip[3],1)
X,Y=np.meshgrid(xs,ys)
xy_pred=np.c_[X.ravel(),Y.ravel()]
z_pred = gp.predict(xy_pred).reshape(X.shape)

# what was the optimized kernel?
print(gp.kernel_)

##
plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
scat=ax.scatter(points[:,0],points[:,1],30,values,cmap=cmap)
ax.axis('equal')

pm=ax.pcolormesh( X,Y,z_pred,cmap=cmap,zorder=-1)
fig.tight_layout()

plt.setp([scat,pm],clim=[-8,4])

ax.axis(utils.expand_xxyy(clip,0.0))



##--------------------------------------------------------------------------------
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


