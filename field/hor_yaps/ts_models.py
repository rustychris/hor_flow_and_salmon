]import os
import glob
from scipy.signal import medfilt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
import track_common
from scipy.interpolate import interp1d

from stompy import utils
from scipy.stats import gaussian_kde
from matplotlib import gridspec, collections, widgets
from stompy.spatial import field
import seaborn as sns
import stompy.plot.cmap as scmap
from stompy.plot import plot_utils
turbo=scmap.load_gradient('turbo.cpt')

import track_common

##

input_path="screen_final"

df=track_common.read_from_folder(input_path)

vel_suffix='_top2m'

df['track'].apply(track_common.calc_velocities,
                  model_u='model_u'+vel_suffix,model_v='model_v'+vel_suffix)
track_common.clip_to_analysis_polygon(df,'track')

## 

# choose a single track for the moment:
track=df.loc['7A96','track']

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

ax.plot( track.x, track.y )
ax.axis('equal')

##

# Make time even.  Focus on position for now.
dt_nom=5.0
dt_steps=np.round( np.diff( track.tnum)/dt_nom )
dt_actual=np.median( np.diff(track.tnum) / dt_steps )
dt_std=np.std( np.diff(track.tnum) / dt_steps )
print("Actual interval: %.2fs +- %.2f"%(dt_actual,dt_std))

track['step']=np.cumsum(np.r_[0,dt_steps]).astype(np.int32)
track_exp=pd.DataFrame()
track_exp['step']=np.arange(expanded_idx.max()+1)
track2=track_exp.merge(right=track,on='step', how='left')

##


data=track2.loc[:,['x','y']]

data.loc[:,'x'] -= data.x.mean()
data.loc[:,'y'] -= data.y.mean()

# Make life a little simpler and just fill
data.loc[:,'x'] = utils.fill_invalid(data.loc[:,'x'].values)
data.loc[:,'y'] = utils.fill_invalid(data.loc[:,'y'].values)

##

# Sort of following https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html

import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels import tsa

data.plot(figsize=(12,8))

## 
dta=data['x']

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)

##

# So what would we expect the form to be...
# ... if using position data?
#    need a trend term

# x[n+1] = x[n] + dt*u[n] + eps
# u[n+1] = u[n] + turn_angle_distro_sample

# Not sure if there is a good way to do that 

arma_mod20 = sm.tsa.ARMA(dta, (2,0)).fit(disp=False)
print(arma_mod20.params)


# in the ARIMA formulation -
#        autoregressive
# x[t] = sum(alpha[1]*x[t-1] ... alpha[p]*x[t-p]) +
#             moving average:
#              
