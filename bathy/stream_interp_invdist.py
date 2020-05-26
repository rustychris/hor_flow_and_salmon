# A basic inverse-distance weighted approach for fitting the
# streamwise coordinates
import pickle
import time

import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.gaussian_process as GP

from pygam import LinearGAM, s, te

from stompy import utils
from stompy.plot import plot_utils
from stompy.model.suntans import sun_driver
## 
pkl_file="all-targets.pkl"

with open(pkl_file,'rb') as fp:
    all_data=pickle.load(fp)

##

targets=np.array([data.target.values for data in all_data])

##

aniso=50 # off-axis scaled up by this factor
nugget=5.0 # along-axis nugget size
power=-2.

scale=np.array([1,aniso])

data=all_data[0]
def interp_invdist(data):
    #data: dataset with target, stream_dist, sample_z
    stream_trans = data.stream_dist.values
    
    valid=np.isfinite(stream_trans[:,0])
    
    dist=utils.mag(stream_trans[valid,:] * scale)
    if len(dist)==0:
        return np.nan
    
    weights=(nugget+dist)**power
    
    z_interp=(weights*data.sample_z.values[valid]).sum() / weights.sum()
    return z_interp

def interp_invdist_plane(data):
    #data: dataset with target, stream_dist, sample_z
    stream_trans = data.stream_dist.values
    sample_z=data.sample_z.values
    
    valid=np.isfinite(stream_trans[:,0])
    
    dist=utils.mag(stream_trans[valid,:] * scale)
    if len(dist)==0:
        return np.nan
    
    weights=(nugget+dist)**power
    
    # Fit a plane to the data
    clf=linear_model.LinearRegression()
    clf.fit(stream_trans[valid,:], sample_z[valid], weights)
    z_pred=clf.predict(np.array([[0,0]]))[0]
    
    return z_pred

def interp_krige(data):
    stream_trans = data.stream_dist.values
    sample_z=data.sample_z.values
    valid=np.isfinite(stream_trans[:,0])

    if valid.sum()==0:
        return np.nan
    
    # gp = GP.GaussianProcessRegressor(kernel=GP.kernels.ConstantKernel(1.0),n_restarts_optimizer=9)
    # kernel=GP.kernels.ConstantKernel(1.0))
    # kernel=None
    # takes 4 seconds to fit.
    #kernel=( GP.kernels.ConstantKernel(1.0,[0.1,10]) * GP.kernels.RBF(1,[0.1,10])
    #         + GP.kernels.WhiteKernel(noise_level=1) )
    # takes 0.4s to fit.
    kernel=GP.kernels.Matern(length_scale=2,nu=1.5)
    # Warnings
    # kernel=GP.kernels.ConstantKernel(1.0,[0.1,10])
    # kernel=GP.kernels.RBF(1.0,[0.1,10])
    
    gp=GP.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=1)
    offset=sample_z[valid].mean()
    start=time.time()
    gp.fit(stream_trans[valid], sample_z[valid]-offset)
    #print("Fit: %fs"%(time.time()-start))
    #import pdb
    #pdb.set_trace()
    z_pred = gp.predict(np.array([[0,0]]))[0]
    return z_pred+offset


def interp_gam(data):
    valid=np.isfinite(data.stream_dist.values[:,0])
    sample_xy=data.sample_xy.values[valid]
    sample_st=data.stream_dist.values[valid]
    sample_z=data.sample_z.values[valid]
    if np.sum(valid)==0:
        return np.nan
    
    gam=LinearGAM( s(0,n_splines=4) + s(1,n_splines=5) + te(0,1,n_splines=4) ).gridsearch(sample_st,sample_z)
    z_pred = gam.predict(np.array([[0,0]]))[0]
    return z_pred
    
meth='gam'

if meth=='invdist':
    meth_pretty=r"Inverse Distance\np=%g, $\alpha$=%g"%(power,aniso)
    params="p%g_aniso%g"%(-power,aniso)
    z_result=[interp_invdist(data) for data in utils.progress(all_data)]
if meth=='invdist_plane':
    meth_pretty="IDW plane\np=%g, $\\alpha$=%g"%(power,aniso)
    params="p%g_aniso%g"%(-power,aniso)
    z_result=[interp_invdist_plane(data) for data in utils.progress(all_data)]
if meth=='krige':
    meth_pretty="Gaussian Process"
    params=""
    z_result=[interp_krige(data) for data in utils.progress(all_data)]
if meth=='gam':
    meth_pretty="GAM"
    params="v2"
    z_result=[interp_gam(data) for data in utils.progress(all_data)]
    
z_result=np.array(z_result)

##
model=sun_driver.SuntansModel.load("../model/suntans/runs/snubby_steady2d_03")
grid=model.grid

##

zoom=(646840.760309839, 648051.3812292194, 4185285.5128993616, 4186236.486084047)

fig=plt.figure(1)
fig.set_size_inches([6,4],forward=True)
fig.clf()
ax=fig.add_subplot(1,1,1)

#plt.scatter(targets[:,0],targets[:,1],20,z_result,cmap='jet')
valid=np.isfinite(z_result)
ccoll=grid.plot_cells(values=z_result,cmap='jet',clim=[-10,2],ax=ax,mask=valid)
missing=grid.plot_cells(color='0.8',ax=ax,mask=~valid)

ax.axis('equal')

plt.colorbar(ccoll,ax=ax,label='Bed elev. (m)')
plt.setp(ax.xaxis,visible=0)
plt.setp(ax.yaxis,visible=0)
plt.setp(ax.spines.values(),visible=0)
plot_utils.scalebar([0.10,0.10],ax=ax,xy_transform=ax.transAxes,L=500.0,dy=0.02,
                    label_txt="m",fractions=[0,0.2,0.4,1.0])
fig.tight_layout()
ax.text(0.05,0.95,meth_pretty,transform=ax.transAxes,va='top')
ax.axis(zoom)


fig.savefig('gridded-%s-%s.png'%(meth,params))

# invdist, and plane look great.
# kriging is total crap.
#  gets better when the mean is removed from the data, as GP assumes mean=0 and
#  does not fit the mean at all.

##

# Take a closer look at kriging in one specific sample

# Choose a simple, straight channel section for testing
# 1050: good, but only has data downstream
# 1150: better.
data=all_data[1150]

valid=np.isfinite(data.stream_dist.values[:,0])
sample_xy=data.sample_xy.values[valid]
sample_st=data.stream_dist.values[valid]
sample_z=data.sample_z.values[valid]


plt.figure(1).clf()
fig,axs=plt.subplots(1,2,num=1)

scat0=axs[0].scatter(sample_xy[:,0],
                     sample_xy[:,1],
                     30,
                     sample_z)
scat1=axs[1].scatter(sample_st[:,0],sample_st[:,1],30,sample_z)
axs[1].plot([0],[0],'ro')
axs[0].plot(data.target[:1],data.target[1:],'ro')

axs[0].axis('equal')


##

bounds=[sample_st[:,0].min(),sample_st[:,0].max(),
        sample_st[:,1].min(),sample_st[:,1].max()]
grid_s=np.linspace(bounds[0],bounds[1],50)
grid_t=np.linspace(bounds[2],bounds[3],50)
S,T=np.meshgrid(grid_s,grid_t)
ST=np.c_[ S.ravel(), T.ravel() ]

# Overall, maybe Kriging is not such a good fit.
# We went to all that effort to retain two dimensions, and kriging
# doesn't really preserve that, at least not in vanilla approaches.
#---
from pygam import LinearGAM, s, te


# gam=LinearGAM( s(0, n_splines=200) + te(3, 1) + s(2)).fit(X, y)
# fit and gridsearch give different answers.  overall gridsearch looks better.
#gam=LinearGAM(n_splines=6).fit(sample_st,sample_z)
#gam=LinearGAM(n_splines=6).gridsearch(sample_st,sample_z)
# same as this:
#gam=LinearGAM( s(0,n_splines=6) + s(1,n_splines=6) ).gridsearch(sample_st,sample_z)

# decent - AIC =2355
# gam=LinearGAM(n_splines=6).gridsearch(sample_st, sample_z)
# pretty nice - but AIC=6260
# gam=LinearGAM( s(0,n_splines=6) + s(1,n_splines=6) + te(0,1,n_splines=5) ).gridsearch(sample_st,sample_z)
# AIC=2005
# gam=LinearGAM( s(0,n_splines=4) + s(1,n_splines=4) + te(0,1,n_splines=4) ).gridsearch(sample_st,sample_z)
# AIC: 2760, but arguably the best looking.
# gam=LinearGAM( s(0,n_splines=4) + s(1,n_splines=5) + te(0,1,n_splines=4) ).gridsearch(sample_st,sample_z)
# AIC=1345
# gam=LinearGAM( s(0,n_splines=4) + s(1,n_splines=4)  ).gridsearch(sample_st,sample_z)
# AIC=1500 -- looks terrible
# gam=LinearGAM( s(0,n_splines=4) ).gridsearch(sample_st,sample_z)
# AIC=1250 -- meh.
# gam=LinearGAM( s(1,n_splines=4) ).gridsearch(sample_st,sample_z)
# AIC 2001
# gam=LinearGAM( te(0,1,n_splines=4) ).gridsearch(sample_st,sample_z)
# AIC 2900
#gam=LinearGAM( te(0,1,n_splines=5) ).gridsearch(sample_st,sample_z)
# 6900, but looks okay
gam=LinearGAM( s(1,n_splines=5) + te(0,1,n_splines=6) ).gridsearch(sample_st,sample_z)

print("AIC: ",gam.statistics_['AIC'])
# gam.summary()

#--
plt.figure(2).clf()
nterms=len(gam.terms)-1 # omit intercept
fig, term_axs = plt.subplots(1,nterms,num=2)
if nterms==1:
    term_axs=[term_axs]
titles = [repr(t) for t in gam.terms]

for i, ax in enumerate(term_axs):
    XX = gam.generate_X_grid(term=i)
    if gam.terms[i].istensor or gam.terms[i].isintercept:
        continue # don't know how to plot those yet
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
    ax.set_title(titles[i])


#--

z_pred = gam.predict(ST)

#---
z_img=z_pred.reshape(S.shape)

axs[1].images=[]
img=axs[1].imshow(z_img,extent=bounds,
                  origin='bottom')
img.set_clim(scat1.get_clim())

