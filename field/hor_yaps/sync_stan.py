"""
Try to emulate yaps sync processing in stan,
with the hope of extending to splines.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import pickle
import pystan
from stompy import utils

## 
data_dir='yaps/full/20180316T0152-20180321T0003'
detections=pd.read_csv(os.path.join(data_dir,'all_detections.csv'))
hydros=pd.read_csv(os.path.join(data_dir,'hydros.csv'),index_col='serial')

yaps_positions=pd.read_csv('yap-positions.csv',index_col='serial')

##
# Update original hydros locations with yap data
for idx,row in yaps_positions.iterrows():
    if idx in hydros.index:
        hydros.loc[idx,'x']=row['yap_x']
        hydros.loc[idx,'y']=row['yap_y']
        hydros.loc[idx,'z']=row['yap_z']

## 

model_file='sync.stan'
model_pkl=model_file+'.pkl'

if utils.is_stale(model_pkl,[model_file]):
    sm = pystan.StanModel(file=model_file)
    with open(model_pkl, 'wb') as fp:
        pickle.dump(sm, fp)
else:
    with open(model_pkl,'rb') as fp:
        sm=pickle.load(fp)

## 

T0=int(detections['epo'].mean())

detections['t_frac']=(detections['epo']-T0) + detections['frac']
data={}

tk_serial='AM8.8'


data['nh']=len(hydros)
data['tk']=hydros.loc[tk_serial,'idx']

detections['hydro']=hydros.loc[detections['serial'],'idx'].values


##

def combine_detects(tag_detects,slop_s=10.0):
    """
    combine received pings with the same tag into one ping when
    separate by less than slop_s
    """
    det_hydros=tag_detects['hydro'].values-1 # to 0-based
    det_t_fracs=tag_detects['t_frac'].values
    toa_rows=[]
    breaks=np.nonzero( np.diff(det_t_fracs)> slop_s)[0]

    breaks=np.r_[ 0,1+breaks,len(det_t_fracs)]
    for b_start,b_stop in zip(breaks[:-1],breaks[1:]):
        slc=slice(b_start,b_stop)

        toa_row=np.nan*np.zeros(len(hydros))
        toa_row[det_hydros[slc]]=det_t_fracs[slc]

        if np.isfinite(toa_row).sum() < b_stop-b_start:
            # now we can do more expensive testing
            mean_t_frac=np.nanmean(det_t_fracs[slc])
            for h in range(len(hydros)):
                if (det_hydros[slc]==h).sum()>1:
                    # t_frac for the colliding detections
                    t_frac_hydro=det_t_fracs[slc][ det_hydros[slc]==h ]
                    # which one is closest to the mean
                    best=np.argmin(np.abs(t_frac_hydro - mean_t_frac))
                    toa_row[h]=t_frac_hydro[best]
                    print("Discarding a ping")

        toa_rows.append(toa_row)
    return np.array(toa_rows)

all_toa_rows=[]
all_sync_tag_idx=[]

sync_tags=hydros['sync_tag'].values
for sync_tag in sync_tags:
    tag_detects=detections[ detections['tag']==sync_tag ].sort_values('t_frac')
    tag_detects=tag_detects.reset_index()
    print(f"tag {sync_tag} with {len(tag_detects)} detections total")
    toa_rows_tag=combine_detects(tag_detects)
    all_toa_rows.append(toa_rows_tag)
    sync_tag_idx=hydros[ hydros['sync_tag']==sync_tag ]['idx'].values[0]
    all_sync_tag_idx.append( sync_tag_idx * np.ones(toa_rows_tag.shape[0]))

##

toa_absolute=np.concatenate( all_toa_rows, axis=0)
sync_tag_idx=np.concatenate( all_sync_tag_idx )

data['np']=len(sync_tag_idx)

data['H']= hydros.loc[:, ['x','y','z']].values
data['sync_tag_idx_vec']=sync_tag_idx.astype(np.int32)

##

# Last thing is the offset and ss periods.
n_offset_idx=data['n_offset_idx']=5
n_ss_idx=data['n_ss_idx']=5

t_min=np.nanmin(toa_absolute)-1
t_max=np.nanmax(toa_absolute)+1

offset_breaks=np.linspace(t_min,t_max,n_offset_idx+1)
ss_breaks    =np.linspace(t_min,t_max,n_ss_idx+1)

toa_absolute_mean=np.nanmean(toa_absolute,axis=1)

offset_idx=np.searchsorted(offset_breaks,toa_absolute_mean)
toa_offset=toa_absolute - offset_breaks[offset_idx,None]

ss_idx=np.searchsorted(ss_breaks,toa_absolute_mean)

data['offset_idx']=offset_idx # already 1-based b/c of t_min
data['ss_idx']=ss_idx         

data['toa_offset'] = toa_offset

# force it to at least try...
data['sigma_toa']=0.0001
##

# HERE
# Even providing pretty reasonable defauls didn't shave
# the log prob that much.
# Seems like there is probably still an error in the data or the stan
# code.
# good news is that even with the default tolerances, it's making
# progress where before it would just bail out without trying.

# It's cranking but seems very slow, and the log_prob is -1e7.
# Almost all of that is coming from the last stanza - eps_toa ~ student_t

init={'ss':1450*np.ones(data['n_ss_idx']),
      'top':np.nanmean(data['toa_offset'],axis=1),
      'offset':np.zeros( (data['nh'],data['n_offset_idx']), np.float64),
      # 'slope1':np.zeros( (data['nh'],data['n_offset_idx']), np.float64),
      # 'slope2':np.zeros( (data['nh'],data['n_offset_idx']), np.float64)
}

data['sigma_toa']=0.001 # coarse fit first?
opt=sm.optimizing(data=data,init=init,verbose=True,iter=50,
                  refresh=10,history_size=20)
##

# A single opt value is 37M.
# Could reduce that maybe by 4x by only keeping nonzero eps_toa.


# Any chance that a local solution is faster?
## 
# Transformed data:
xdata=dict(data)
dist_mat=np.zeros( (data['nh'],data['nh']), np.float64)
for h1 in range(data['nh']):
    for h2 in range(data['nh']):
        dist_mat[h1,h2]=utils.dist(data['H'][h1],data['H'][h2])
xdata['dist_mat']=dist_mat
xdata['off_mask']=np.arange(xdata['nh'])!=(xdata['tk']-1)
off_mask=xdata['off_mask']
## 
dist_mat=xdata['dist_mat']
sync_tag_idx_vec=xdata['sync_tag_idx_vec']-1
ss_idx=xdata['ss_idx']-1
offset_idx=xdata['offset_idx']-1
toa_offset=xdata['toa_offset']
valid=np.isfinite(toa_offset)

def pack_parameters(kw):
    return np.r_[ kw['offset'][off_mask,:].ravel(), kw['ss'] ]

def unpack_parameters(vec):
    i=0
    kw={}
    kw['offset']=np.zeros( (data['nh'],data['n_offset_idx']), np.float64)
    kw['offset'][off_mask,:] = (vec[i:i+(data['nh']-1)*data['n_offset_idx']].reshape( (data['nh']-1,data['n_offset_idx']) ))
    i+=(data['nh']-1)*data['n_offset_idx']
    kw['ss']=vec[i:i+data['n_ss_idx']]
    i+=data['n_ss_idx']
    assert i==len(vec)
    return kw
                 
def my_log_prob(vec,out=False):
    kw=unpack_parameters(vec)

    # [np,nh] giving transit time
    transit_times = dist_mat[sync_tag_idx_vec,:] / kw['ss'][ss_idx,None]
    # [np,nh] giving adjusted time of arrival
    toa_adjusted = xdata['toa_offset'] + kw['offset'][:,offset_idx].T

    top_estimate = toa_adjusted - transit_times

    # minimize variance of top estimates
    error=np.nanvar(top_estimate,axis=1).sum()

    # Now include some priors:
    ss_prior = ((kw['ss']-1450)**2 / (30*30)).sum()
    off_prior = (kw['offset']**2 / (10*10)).sum()

    post= error + ss_prior + off_prior
    return post

init=dict(offset=np.zeros( (xdata['nh'],xdata['n_offset_idx']) ),
          ss=1463*np.ones(xdata['n_ss_idx']))
vec=pack_parameters(init)
p=my_log_prob(vec)

##

from scipy.optimize import fmin,fmin_powell

def update(xk):
    p=unpack_parameters(xk)
    print(p['ss'])
    print(p['offset'])
    
# down to 13 parameters.
# soundspeed is getting pretty consistent.
# starting nll: 105992
# With the bug fix, it's now getting some very reasonable looking
# offsets.
best=fmin_powell(my_log_prob,pack_parameters(init),
                 callback=update)

##

py_opt=unpack_parameters(best)
plt.figure(2).clf()
fig,axs=plt.subplots(3,1,num=2)

ss_mid=0.5*(ss_breaks[:-1] + ss_breaks[1:])
axs[0].plot( ss_mid, py_opt['ss'],marker='o', label='ss')

offset_mid=0.5*( offset_breaks[:-1] + offset_breaks[1:])

for h in range(data['nh']):
    axs[1].plot( offset_mid, py_opt['offset'][h,:],marker='o', label="H=%d"%h)
    

# [np,nh] giving transit time
transit_times = dist_mat[sync_tag_idx_vec,:] / py_opt['ss'][ss_idx,None]
# [np,nh] giving adjusted time of arrival
toa_adjusted = xdata['toa_offset'] + py_opt['offset'][:,offset_idx].T
top_estimate = toa_adjusted - transit_times
errors=np.nanvar(top_estimate,axis=1)

axs[2].hist(np.sqrt(errors),bins=np.linspace(0.0,0.06,100))
axs[2].set_xlabel('Time-of-flight error (s)')

p=my_log_prob(best)


##

# lower tol_rel_grad does force it to try harder...
opt2=sm.optimizing(data=data,init=opt,tol_rel_grad=1e6)

vis_results(data,opt2)

##
# This goes for a while and maxes out n iterations
data['sigma_toa']=0.005

opt3=sm.optimizing(data=data,init=opt2,tol_rel_grad=1e5)

vis_results(data,opt3)

##

# This ends after a bit with relative gradient below tolerance
# What about Newton?  fails with error
opt4=sm.optimizing(data=data,init=opt3,tol_rel_grad=1e5)

##

def vis_results(data,opt):
    # Visualize the results:
    plt.figure(1).clf()
    fig,axs=plt.subplots(3,1,num=1)

    # Distribution of eps_toa:
    ping_valid=np.isfinite(data['toa_offset'])
    axs[0].hist( opt['eps_toa'][ping_valid] )
    axs[0].set_xlabel('$\epsilon$ TOA (s)')

    axs[1].hist( opt['ss'] )
    axs[1].set_xlabel('Soundspeed')

    #for h in range(
    # axs[2].
