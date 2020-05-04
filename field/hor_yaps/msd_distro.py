"""
Plot all of the presumed fish tracks as progressive-swim-vector
and take a look at d msd / dt for superdiffusion.
"""

import os
import glob
from scipy.signal import medfilt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import track_common
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa import stattools
from stompy import utils
import stompy.plot.cmap as scmap
import seaborn as sns
from stompy import nanpsd

turbo=scmap.load_gradient('turbo.cpt')
##
def unwrap(angles):
    deltas=(np.diff(angles) + np.pi)%(2*np.pi) - np.pi
    return np.cumsum( np.r_[angles[0],deltas] )

input_path="screen_final"
fig_dir=os.path.join(input_path,'figs-20200430')
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

df=track_common.read_from_folder(input_path)

def add_more_swim_data(track):
    swim_hdg_rel_uw=np.r_[ unwrap(track['swim_hdg_rel'].values[:-1]),
                           np.nan ]
    track['swim_hdg_rel_uw']=swim_hdg_rel_uw
    track['swim_speed']=utils.mag(track.loc[:,['swim_u','swim_v']].values)

    # and the swim-only track
    dt=np.diff(track['tnum'].values)
    dx=np.cumsum( dt*track['swim_urel'].values[:-1])
    dy=np.cumsum( dt*track['swim_vrel'].values[:-1])
    track['swim_x']=np.r_[ 0, dx ]
    track['swim_y']=np.r_[ 0, dy ]

df['track'].apply( add_more_swim_data )

##

# All tracks
num=20
fig=plt.figure(num)
fig.clf()
g=gridspec.GridSpec(3,5)
ax_swim=fig.add_subplot(g[:,0])

ax_msd_x=fig.add_subplot(g[0,1:3])
ax_msd_y=fig.add_subplot(g[1,1:3])
ax_msd  =fig.add_subplot(g[2,1:3])
ax_box_x=fig.add_subplot(g[0,3:])
ax_box_y=fig.add_subplot(g[1,3:])
ax_box  =fig.add_subplot(g[2,3:])


for ti in range(len(df)):
    track=df['track'][ti]

    # Plot with the same orientation as the swim speed distribution
    # figures
    ax_swim.plot(track['swim_y'], track['swim_x'], color=turbo(ti/float(len(df)-1)))

ax_swim.axis('equal')

ax_swim.axhline(0,color='k',lw=0.5)
ax_swim.axvline(0,color='k',lw=0.5)


msd_rows=[]

for ti in range(len(df)):
    track=df['track'][ti]
    swim_msd_x=(track['swim_x']-track['swim_x'].values[0])**2
    swim_msd_y=(track['swim_y']-track['swim_y'].values[0])**2
    swim_msd  =swim_msd_x+swim_msd_y
    t=track.tnum.values-track.tnum.values[0]
    ax_msd_x.plot(t,swim_msd_x,color=turbo(ti/float(len(df)-1)))
    ax_msd_y.plot(t,swim_msd_y,color=turbo(ti/float(len(df)-1)))
    ax_msd.plot(t,  swim_msd,color=turbo(ti/float(len(df)-1)))

    if t[-1]>600:
        sel=(t<=600)
        msd_df=pd.DataFrame()
        msd_df['t']=t[sel]
        msd_df['msd_x']=swim_msd_x[sel]
        msd_df['msd_y']=swim_msd_y[sel]
        msd_df['msd']=swim_msd[sel]
        msd_rows.append( msd_df )
    
ax_msd_x.text(0.01,0.9,'$x^2$',transform=ax_msd_x.transAxes)
ax_msd_y.text(0.01,0.9,'$y^2$',transform=ax_msd_y.transAxes)
ax_msd.text(0.01,0.9,'$x^2+y^2$',transform=ax_msd.transAxes)

ax_msd_x.yaxis.set_visible(0)
ax_msd_y.yaxis.set_visible(0)
ax_msd.yaxis.set_visible(0)

msd_data=pd.concat( msd_rows )

bins=np.linspace(0,600,31)
msd_data['t_bin'] = np.searchsorted(bins,msd_data.t.values)

sns.boxplot(x='t_bin',y='msd_x',data=msd_data,ax=ax_box_x)
sns.boxplot(x='t_bin',y='msd_y',data=msd_data,ax=ax_box_y)
sns.boxplot(x='t_bin',y='msd',data=msd_data,ax=ax_box)

fig.tight_layout()
fig.savefig(os.path.join(fig_dir,'swim_progressive_vector.png'))
