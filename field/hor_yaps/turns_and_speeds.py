import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import medfilt
import statsmodels.api as sm
import numpy as np
import six
import os
import track_common
six.moves.reload_module(track_common)

import seaborn as sns
from stompy import memoize, utils
import pandas as pd
from stompy.io.local import cdec
from scipy.stats.kde import gaussian_kde
from scipy import signal 
##

vel_suffix='_top2m'

fig_dir="fig_turns-20201010"+vel_suffix
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

df_start=track_common.read_from_folder('screen_final')

df_start['track'].apply(track_common.calc_velocities,
                        model_u='model_u'+vel_suffix,
                        model_v='model_v'+vel_suffix)

if 0: # For the moment don't do the extra clipping
    track_common.clip_to_analysis_polygon(df_start,'track')

##

# Drop some irrelevant columns
cols=['x', 'y', 'tnum', 'x_sd', 'y_sd', 'x_m', 'y_m', 'tnum_m',
      'model_u_top2m', 'model_v_top2m',
      'groundspeed', 'ground_u', 'ground_v', 'model_hdg',
      'ground_hdg', 'ground_urel', 'ground_vrel', 'tnum_end', 'x_end',
      'y_end', 'swim_u', 'swim_v', 'swim_hdg', 'swim_hdg_rel', 'swim_urel',
      'swim_vrel' ]

## 

for stride in [1,2,3,4,5]:
    fig_num=1+stride
    accum=[]

    for track in df_start['track']:
        track=track.loc[:,cols]
        track_exp=track_common.expand_to_regular(track,track.tnum)

        for off in range(stride):
            sub_track=track_exp.iloc[off::stride].copy()
            track_common.calc_velocities( sub_track,'model_u_top2m','model_v_top2m' )
            sub_track['swim_speed']=np.sqrt( sub_track.swim_urel**2 + sub_track.swim_vrel**2)
            dspeed_rel=np.diff(sub_track['swim_speed'])
            dhdg_rel=np.diff(sub_track['swim_hdg_rel'])
            # force to [-pi,pi]
            dhdg_rel=(dhdg_rel+np.pi)%(2*np.pi) - np.pi
            sub_track['dspeed_rel']=np.r_[dspeed_rel, np.nan]
            sub_track['dhdg_rel']  =np.r_[dhdg_rel, np.nan]
            sel=np.isfinite(sub_track['dhdg_rel'])
            accum.append(sub_track[sel])

    strided=pd.concat(accum)

    from scipy.stats import cauchy

    plt.figure(fig_num).clf()
    fig,axs=plt.subplots(2,1,num=fig_num)
    axs[0].hist(strided['dspeed_rel'],bins=np.linspace(-0.5,0.5,100),
                density=True)
    iqrs=np.percentile( strided['dspeed_rel'],[25,75])
    axs[0].axvline( iqrs[0], color='k')
    axs[0].axvline( iqrs[1], color='k')

    axs[1].hist(strided['dhdg_rel'],bins=np.linspace(-3.0,3.0,100),
                density=True)
    iqrs=np.percentile( strided['dhdg_rel'],[25,75])
    axs[1].axvline( iqrs[0], color='k')
    axs[1].axvline( iqrs[1], color='k')

    axs[0].set_title(f'Stride: {stride}')

    res=cauchy.fit(strided['dspeed_rel'])
    loc,scale=res
    samp=np.linspace(-0.5,0.5,100)
    pdf=cauchy.pdf(samp, *res)
    axs[0].plot(samp,pdf,color='tab:orange')
    axs[0].text(0.05,0.85,f"Cauchy(loc={loc:.3f}, scale={scale:.3f})",
                transform=axs[0].transAxes)


    res=cauchy.fit(strided['dhdg_rel'])
    loc,scale=res
    samp=np.linspace(-np.pi,np.pi,100)
    pdf=cauchy.pdf(samp, *res)
    axs[1].plot(samp,pdf,color='tab:orange')
    axs[1].text(0.05,0.85,f"Cauchy(loc={loc:.3f}, scale={scale:.3f})",
                transform=axs[1].transAxes)

##

# HERE
#  For one, I have't removed rheotaxis.  Probably not great.
#  
#   Longer strides make change in speed smaller...
#   
#  scale    dspeed  dhdg
# Stride  1  0.055  0.35
#         2  0.047  0.35
#         3  0.044  0.36 
#         4  0.042  0.37
#         5  0.041  0.38

# The question is how much of this variation is due to random noise
# and how much is due to jerky fish movements.
# Could explore a model like
# True position of fish f at time step n:
# speed_stoch(f,n) = speed_stoch(f,n-1) + random_dspd|speed_stoch(f,n-1)
# heading(f,n)     = heading(f,n-1) + random_dhdg
# swim_rel = u_rheo(f) + {cos(heading(f,n)), sin(heading(f,n))}*speed_stoch(f,n)
# vector velocity in E-N coordinates
# vel = swim_rel*R(hydro_theta(f,n)) + hydro_vel(f,n)
# True fish position:
# X(f,n) = X(f,n-1) + vel*dt
# Detected fish position
#   Xd(f,n) = X(f,n) + norm(0,eps)

# The task is to estimate eps, u_rheo(f), parameters for random_dhdg, random_dspd
# Does it seem like this is well posed?
#  
# How would you write this as a stan model?
# There is hydrodynamic noise
# 
# 
##    

track=df_start['track'][4].loc[:,cols]
track_exp=track_common.expand_to_regular(track,track.tnum)

fig=plt.figure(1)
fig.clf()
from matplotlib import gridspec
gs=gridspec.GridSpec(3,1)
ax=fig.add_subplot(gs[0,:])
ax_t=fig.add_subplot(gs[1,:])
ax_hdg=fig.add_subplot(gs[2,:],sharex=ax_t)

ax.plot( track_exp.x, track_exp.y, 'go')
ax.set_adjustable('datalim')
ax.set_aspect(1.0)

swim_mag=np.sqrt( track_exp.swim_urel**2 + track_exp.swim_vrel**2)
ax_t.plot( track_exp.tnum, swim_mag,'go')

ax_hdg.plot( track_exp.tnum, track_exp.swim_hdg_rel,'go')
ax_hdg.axis(ymin=-np.pi,ymax=np.pi)

# Is it possible that the fish really do just lurch and hold?
#

##

# So Ed calculates dspeed and speed, fits a distribution to
# dspeed, and is concerned that it is of the same order of
# magnitude as speed.

