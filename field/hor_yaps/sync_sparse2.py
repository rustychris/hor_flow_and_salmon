"""
Try to emulate yaps sync processing in stan,
with the hope of extending to splines.

This version moves to linear interpolation with a direct matrix-based 
solution
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

# First 12 hours work pretty well
# t_sel_min=all_detections['epo'].min()
# t_sel_max=t_sel_min+12*3600
# detections=all_detections[ (all_detections['epo']>=t_sel_min)&(all_detections['epo']<t_sel_max) ]

# Full period is a bit slower, a memory hog, and results are not good
# detections=all_detections

# Second 12 hours?  had to add some code to get rid of bad pings.
# with that done, it now gets the same 0.4ms RMSE as the 1st 12 hours.

# 36:48: looks like some sparse coverage on 2 hydros.
# 72:84: same

# Next 12 hours:
t_sel_min=all_detections['epo'].min() + 12*3600
t_sel_max=t_sel_min+12*3600
detections=all_detections[ (all_detections['epo']>=t_sel_min)&(all_detections['epo']<t_sel_max) ].copy()

T0=int(detections['epo'].mean())

detections['t_frac']=(detections['epo']-T0) + detections['frac']
data={}

tk_serial='AM8.8'

data['nh']=len(hydros)
data['tk']=hydros.loc[tk_serial,'idx']

detections['hydro']=hydros.loc[detections['serial'],'idx'].values

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

# --

toa_absolute=np.concatenate( all_toa_rows, axis=0)
sync_tag_idx=np.concatenate( all_sync_tag_idx )

data['np']=len(sync_tag_idx)
data['H']= hydros.loc[:, ['x','y','z']].values
data['sync_tag_idx_vec']=sync_tag_idx.astype(np.int32)

#--

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

# --

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

dist_mat=xdata['dist_mat']
sync_tag_idx_vec=xdata['sync_tag_idx_vec']-1
ss_idx0=xdata['ss_idx']-1
offset_idx0=xdata['offset_idx']-1
toa_offset=xdata['toa_offset']
valid=np.isfinite(toa_offset)

# Attempt a direct matrix solve
Mrows=[]
bvals=[]

nh=xdata['nh']
n_off=xdata['n_offset_idx']
# piecewise linear -- n_off intervals => n_off+1 values
ncols=((n_off+1) * nh) + (xdata['n_ss_idx'] + 1)

row_pings=[]

for p in range(xdata['np']):
    off_idx = xdata['offset_idx'][p]-1
    ss_idx = xdata['ss_idx'][p]-1
    tag_idx=xdata['sync_tag_idx_vec'][p]-1

    ss_alpha=mean_toa_offset[p]/(ss_breaks[ss_idx+1] - ss_breaks[ss_idx])
    off_alpha=( mean_toa_offset[p]
                /
                (offset_breaks[off_idx+1] - offset_breaks[off_idx]) )
    
    hi = np.nonzero( np.isfinite(xdata['toa_offset'][p,:] ))[0]
    h1=hi[0]
    for h2 in hi[1:]:
        row=np.zeros( ncols, np.float64)
        row[ (n_off+1)*h1 + off_idx]   = -(1.0-off_alpha)
        row[ (n_off+1)*h1 + off_idx+1] = -(off_alpha)
        row[ (n_off+1)*h2 + off_idx] = (1-off_alpha)
        row[ (n_off+1)*h2 + off_idx+1] = off_alpha

        # soundspeed rows have inverse, so they are second / meter
        row[ nh*(n_off+1) + ss_idx] = (xdata['dist_mat'][h1,tag_idx] - xdata['dist_mat'][h2,tag_idx]) * (1-ss_alpha)
        row[ nh*(n_off+1) + ss_idx + 1] = (xdata['dist_mat'][h1,tag_idx] - xdata['dist_mat'][h2,tag_idx]) * ss_alpha
        
        Mrows.append(row)
        bvals.append( xdata['toa_offset'][p,h1] - xdata['toa_offset'][p,h2] )
        row_pings.append(p)
# -- 

# so a row represents the difference in time-of-flight for a ping
# from tag_idx to get to h1 vs. h2

M=np.array(Mrows)
b=np.array(bvals)
row_pings=np.array(row_pings)

# That includes values for the timekeeper
# drop those columns.
sel=np.ones(M.shape[1],np.bool8)
tk=xdata['tk']-1
sel[tk*(n_off+1):(tk+1)*(n_off+1)]=False
Mslim=M[:,sel]

init=np.zeros(Mslim.shape[1])
init[-(xdata['n_ss_idx']+1):]=1./1450

bad_pings=[]

rmse=100

while len(bad_pings)<200:
    valid=np.ones(Mslim.shape[0],np.bool8)
    valid[bad_pings]=False

    Mslim_nobad=Mslim[valid,:]

    # This very quickly gets down to 3.15ms.
    # powell gets there, but slower.
    soln,res,rank,sing=np.linalg.lstsq(Mslim[valid,:],
                                       b[valid],
                                       rcond=-1)

    Merrors=np.zeros(Mslim.shape[0],np.float64)
    Merrors[valid]=Mslim[valid,:].dot(soln) - b[valid]

    rmse=np.sqrt( (Merrors**2).mean() )
    print("RMSE: ",rmse)
    if rmse<0.001:
        print("Good enough")
        break

    # Mark the worst ping as bad.
    bad_pings.append( np.argmax( np.abs(Merrors) ) )
    print("Worst ping row had error=",Merrors[bad_pings[-1]])

    print(bad_pings)
else:
    raise Exception("Failed to get rid of enough bad pings to get a good RMSE")

#--

# repack that into the form expected by previous code, to get an apples to apples
# comparison.
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
                 
def my_log_prob_lin(vec,out=False,valid=slice(None)):
    kw=unpack_parameters_lin(vec)

    # Linear interpolation
    # this only works when ss_breaks and offset_breaks are the same!
    ss_alpha= mean_toa_offset/(ss_breaks[ss_idx0+1] - ss_breaks[ss_idx0])
    ss_vals=(1-ss_alpha)*kw['ss'][ss_idx0] + ss_alpha*kw['ss'][ss_idx0+1]
        
    # [np,nh] giving transit time
    transit_times = dist_mat[sync_tag_idx_vec,:] / ss_vals[:,None]

    # Linear
    off_alpha=( mean_toa_offset
                /
                (offset_breaks[offset_idx0+1] - offset_breaks[offset_idx0]) )
    
    offset_vals=( (1-off_alpha)*kw['offset'][:,offset_idx0]
                  +
                  off_alpha*kw['offset'][:,offset_idx0+1] )
    
    # [np,nh] giving adjusted time of arrival
    toa_adjusted = xdata['toa_offset'] + offset_vals.T

    top_estimate = toa_adjusted - transit_times

    # minimize variance of top estimates
    errors=np.nanvar(top_estimate,axis=1)[valid]
    error=errors.sum() / (0.001**2)

    # Now include some priors:
    ss_prior = ((kw['ss']-1450)**2 / (30*30)).sum()
    off_prior = (kw['offset']**2 / (10*10)).sum()

    post= error + ss_prior + off_prior
    if out:
        return dict(post=post,rms_error=np.sqrt(errors.mean()),
                    nll_ss=ss_prior,nll_off=off_prior,nll_error=error,
                    errors=errors,
                    offset_vals=offset_vals,
                    ss_vals=ss_vals)
    else:
        return post

best=soln.copy()
best[-(xdata['n_ss_idx']+1):] = 1./best[-(xdata['n_ss_idx']+1):] 

p=my_log_prob_lin(best)

#

if 0:
    # This includes pings with only one receiver, which
    # aren't useful.
    valid_pings=np.ones(len(mean_toa_offset), np.bool8 )
    valid_pings[ row_pings[~valid] ] = False
if 1:
    # This is fairer -- but departs from previous scripts
    valid_pings=np.zeros(len(mean_toa_offset), np.bool8 )
    valid_pings[ row_pings[valid] ] = True
    valid_pings[ row_pings[~valid] ] = False

py_opt=unpack_parameters_lin(best)
calcs=my_log_prob_lin(best,out=True,valid=valid_pings)

plt.figure(2).clf()
fig,axs=plt.subplots(3,1,num=2)

order=np.argsort(toa_absolute_mean)
axs[0].plot( toa_absolute_mean[order], calcs['ss_vals'][order], 'g',ms=1)

for h in range(data['nh']):
    axs[1].plot( toa_absolute_mean[order], calcs['offset_vals'][h,order],label="H=%d"%h)

axs[2].hist(np.sqrt(calcs['errors']),bins=np.logspace(-5,-2,100))
axs[2].set_xlabel('Time-of-flight error (s)')
axs[2].set_xscale('log')

axs[2].text(0.9,0.95,
            "\n".join([ "nll: %.3e"%calcs['post'],
                        "  error: %.3e"%calcs['nll_error'],
                        "  offset: %.3e"%calcs['nll_off'],
                        "  ss: %.3e"%calcs['nll_ss'],
                        "rmse: %.4f"%calcs['rms_error']]),
            transform=axs[2].transAxes,va='top',ha='right')

##

# HERE:
#  The next question is whether this is comparable to the sync that yaps
#  does.
#  How best to compare them?
#  Maybe the easiest is to run the yaps sync for one of these 12 hour chunks,
#   and save the offsets,slope1,slope2 to netcdf or similar.
#   then I can use the sync_powell_yaps.py to do the comparison.


# Or... I could apply these shifts to detections, write that out, 
# process in yaps with 1 offset idx and 1 ss idx ?

# If it is possible to translate yaps_track.h into python/stan/scipy,
# I'm more inclined to bring the yaps sync output into python.
# If that's a rabbit-hole-in-waiting, better to push this sync data
# to R.

# The yaps code currently doesn't allow for linearly interpolated SS.
# 
# It's a bit involved -- better to push sync.  In fact just push the sync parameters.
# And since this is a subset of the 
