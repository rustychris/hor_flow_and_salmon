"""
Explore a manually constructed model for segmenting tracks
"""
import os
import glob
from scipy.signal import medfilt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import track_common
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa import stattools
from stompy import utils
import seaborn as sns
from stompy import nanpsd

##
def unwrap(angles):
    deltas=(np.diff(angles) + np.pi)%(2*np.pi) - np.pi
    return np.cumsum( np.r_[angles[0],deltas] )

#input_path="screen_final"
input_path="with_nonsmolt"
fig_dir=os.path.join(input_path,'figs-20200502')
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

track=df.loc['7A96','track'].copy()

segs=track[:-1]

# variables to cluster on:
from sklearn import cluster 

if 0: 
    # standardize
    state=segs.loc[:, ['swim_hdg_rel','swim_speed','tnum_m'] ].values
    state=(state-state.mean(axis=0)) / state.std(axis=0)
    kmeans=cluster.KMeans(n_clusters=5).fit(state)
    labels=kmeans.labels_
if 1:
    # standardize
    state=segs.loc[:, ['swim_hdg_rel','swim_speed','tnum_m'] ].values
    state=(state-state.mean(axis=0)) / state.std(axis=0)
    spectral=cluster.SpectralClustering(n_clusters=6).fit(state)
    labels=spectral.labels_

num=20
plt.figure(num).clf()
fig,(ax_geo,ax_swim)=plt.subplots(2,1,num=num)

ax_geo.scatter( segs['x_m'], segs['y_m'],20,labels,cmap='jet')
ax_swim.scatter( segs['swim_x'], segs['swim_y'],20,labels,cmap='jet')

ax_geo.axis('equal')
ax_swim.axis('equal')

