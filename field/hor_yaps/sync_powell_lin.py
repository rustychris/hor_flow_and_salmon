"""
Try to emulate yaps sync processing in stan,
with the hope of extending to splines.

This version moves to linear interpolation
"""

import matplotlib.pyplot as plt
from matplotlib import collections
import pandas as pd
import os
import numpy as np
from stompy import utils
from scipy.interpolate import UnivariateSpline

## 
data_dir='yaps/full/20180316T0152-20180321T0003'
all_detections=pd.read_csv(os.path.join(data_dir,'all_detections.csv'))
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

# Look at just 12 hours
t_sel_min=all_detections['epo'].min()
t_sel_max=t_sel_min+12*3600
detections=all_detections[ (all_detections['epo']>=t_sel_min)&(all_detections['epo']<t_sel_max) ]

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
n_offset_idx=data['n_offset_idx']=4
n_ss_idx=data['n_ss_idx']=4

t_min=np.nanmin(toa_absolute)-1
t_max=np.nanmax(toa_absolute)+1

offset_breaks=np.linspace(t_min,t_max,n_offset_idx+1)
ss_breaks    =np.linspace(t_min,t_max,n_ss_idx+1)

toa_absolute_mean=np.nanmean(toa_absolute,axis=1)

offset_idx=np.searchsorted(offset_breaks,toa_absolute_mean)
toa_offset=toa_absolute - offset_breaks[offset_idx-1,None]

ss_idx=np.searchsorted(ss_breaks,toa_absolute_mean)

data['offset_idx']=offset_idx # already 1-based b/c of t_min
data['ss_idx']=ss_idx         

data['toa_offset'] = toa_offset

data['sigma_toa']=0.0001

## 
# Transformed data:
xdata=dict(data)
dist_mat=np.zeros( (data['nh'],data['nh']), np.float64)
for h1 in range(data['nh']):
    for h2 in range(data['nh']):
        dist_mat[h1,h2]=utils.dist(data['H'][h1],data['H'][h2])
xdata['dist_mat']=dist_mat
xdata['off_mask']=np.arange(xdata['nh'])!=(xdata['tk']-1)
xdata['mean_toa_offset']=np.nanmean(xdata['toa_offset'],axis=1)

off_mask=xdata['off_mask']
mean_toa_offset=xdata['mean_toa_offset']
## 
dist_mat=xdata['dist_mat']
sync_tag_idx_vec=xdata['sync_tag_idx_vec']-1
ss_idx0=xdata['ss_idx']-1
offset_idx0=xdata['offset_idx']-1
toa_offset=xdata['toa_offset']
valid=np.isfinite(toa_offset)

def pack_parameters(kw):
    return np.r_[ kw['offset'][off_mask,:].ravel(), kw['ss'] ]

def unpack_parameters_lin(vec):
    i=0
    kw={}
    nh=data['nh'] ; no=data['n_offset_idx']
    kw['offset']=np.zeros( (nh,no+1), np.float64)
    kw['offset'][off_mask,:] = vec[i:i+(nh-1)*(no+1)].reshape( (nh-1,no+1) )
    i+=(nh-1)*(no+1)
    kw['ss']=vec[i:i+data['n_ss_idx']+1]
    i+=data['n_ss_idx']+1
    assert i==len(vec)
    return kw
                 
def my_log_prob_lin(vec,out=False):
    kw=unpack_parameters_lin(vec)

    if 0: # Linear interpolation
        # this only works when ss_breaks and offset_breaks are the same!
        ss_alpha= mean_toa_offset/(ss_breaks[ss_idx0+1] - ss_breaks[ss_idx0])
        ss_vals=(1-ss_alpha)*kw['ss'][ss_idx0] + ss_alpha*kw['ss'][ss_idx0+1]
    else:
        spl=UnivariateSpline( ss_breaks,kw['ss'] )
        ss_vals=spl(toa_absolute_mean)
        
    # [np,nh] giving transit time
    transit_times = dist_mat[sync_tag_idx_vec,:] / ss_vals[:,None]

    if 0: # Linear
        off_alpha=( mean_toa_offset
                    /
                    (offset_breaks[offset_idx0+1] - offset_breaks[offset_idx0]) )

        offset_vals=( (1-off_alpha)*kw['offset'][:,offset_idx0]
                      +
                      off_alpha*kw['offset'][:,offset_idx0+1] )
    else:
        # Splines
        offset_vals=np.zeros( [xdata['nh'],xdata['np']],np.float64)
        for h in range(xdata['nh']):
            spl=UnivariateSpline( offset_breaks, kw['offset'][h,:] )
            offset_vals[h,:]=spl(toa_absolute_mean)
    
    # [np,nh] giving adjusted time of arrival
    toa_adjusted = xdata['toa_offset'] + offset_vals.T

    top_estimate = toa_adjusted - transit_times

    # minimize variance of top estimates
    errors=np.nanvar(top_estimate,axis=1)
    error=errors.sum() / (0.001**2)

    # Now include some priors:
    ss_prior = ((kw['ss']-1450)**2 / (30*30)).sum()
    off_prior = (kw['offset']**2 / (10*10)).sum()

    post= error + ss_prior + off_prior
    if out:
        return dict(post=post,rms_error=np.sqrt(errors.mean()),
                    nll_ss=ss_prior,nll_off=off_prior,nll_error=error)
    else:
        return post

init=dict(offset=np.zeros( (xdata['nh'],1+xdata['n_offset_idx']) ),
          ss=1463*np.ones(1+xdata['n_ss_idx']))

vec=pack_parameters(init)
p=my_log_prob_lin(vec)

##

from scipy.optimize import fmin,fmin_powell

def update(xk):
    p=unpack_parameters_lin(xk)
    print(p['ss'])
    print(p['offset'])
    vals=my_log_prob_lin(xk,out=True)
    print("rms error: %.4f"%vals['rms_error'])
    print("----")
    
best=fmin_powell(my_log_prob_lin,pack_parameters(init),
                 callback=update)

##

# No improvement.
polish=fmin_powell(my_log_prob_lin,best,
                   callback=update)

##

py_opt=unpack_parameters_lin(best)
plt.figure(2).clf()
fig,axs=plt.subplots(3,1,num=2)

def plot_recon_time_series(ax,breaks,values,**kw):
    return ax.plot( breaks, values, marker='o', ms=3, **kw)
    
plot_recon_time_series(axs[0],ss_breaks,py_opt['ss'],label='ss')

for h in range(data['nh']):
    plot_recon_time_series(axs[1],offset_breaks,py_opt['offset'][h,:],label="H=%d"%h)

# [np,nh] giving transit time
transit_times = dist_mat[sync_tag_idx_vec,:] / py_opt['ss'][ss_idx,None]
# [np,nh] giving adjusted time of arrival
toa_adjusted = xdata['toa_offset'] + py_opt['offset'][:,offset_idx].T
top_estimate = toa_adjusted - transit_times
errors=np.nanvar(top_estimate,axis=1)

axs[2].hist(np.sqrt(errors),bins=np.linspace(0.0,0.06,100))
axs[2].set_xlabel('Time-of-flight error (s)')

calcs=my_log_prob_lin(best,out=True)

axs[2].text(0.2,0.95,
            "\n".join([ "nll: %.3e"%calcs['post'],
                        "  error: %.3e"%calcs['nll_error'],
                        "  offset: %.3e"%calcs['nll_off'],
                        "  ss: %.3e"%calcs['nll_ss'],
                        "rmse: %.4f"%calcs['rms_error']]),
            transform=axs[2].transAxes,va='top')

# Piecewise constant, 5 days:
# num ss=1, num off=1 => rms error 0.0629, nll 688
# num ss=2, num off=2 => rms error 0.0436, nll 333
# num ss=4, num off=4 => rms error 0.0376, nll 249

# Piecewise constant, 12 hours:
# With ss=4, off=4 for just 12 hours, gets down to rms error 0.0022
# With the change in cost function this might be improved now.

# Linear, 12 hours:
#   rms error ends at 0.0004 - not bad.
# ends on SS=[1469.01291307 1467.93917901 1467.9917421  1467.17359207 1465.87347973]

# Spline for SS, 12 hours:
#   ends on SS=[1469.4629582  1468.37919211 1467.2371322  1467.90896206 1466.03909124]
# rmse still 0.0004.

##

# What about splines?

# currently my cost function takes about 6.2ms
# with splines it takes 6.5ms

