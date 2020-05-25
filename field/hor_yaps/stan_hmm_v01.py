"""
Explore a HMM in Stan for identifying behaviors and segmenting tracks
"""
import os
import glob
from scipy.signal import medfilt
from stompy import filters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec,cm
import track_common
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa import stattools
from stompy import utils
import seaborn as sns
from scipy import stats
from stompy import nanpsd
import pystan
import pickle
##
input_path="with_nonsmolt"
fig_dir=os.path.join(input_path,'figs-20200502')
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

df=track_common.read_from_folder(input_path)

def add_more_swim_data(track):
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

data=dict(K=3)

# seems this should be ~25, but that tends to overpower heading
data['spd_beta']=4.0*np.ones(data['K']) # rate for speed gamma distros. would like to fit this, too.
data['spd_beta'][0]=30.0
data['spd_beta'][-1]=30.0
data['alpha']=5*np.eye(data['K']) + 1.

import six
six.moves.reload_module(track_common)
if 1: # expanded, lowpassed data
    expand=track_common.expand_to_regular(track,track['tnum'],dt_nom=5.0,
                                          fields=['tnum','swim_x','swim_y','x','y',
                                                  'model_u_surf','model_v_surf'],
                                          fill='interp')
    
    t=expand['tnum'].values
    swim_x=expand['swim_x'].values
    swim_y=expand['swim_y'].values
    geo_x=expand['x'].values
    geo_y=expand['y'].values
    hyd_u=expand['model_u_surf'].values
    hyd_v=expand['model_v_surf'].values

    winsize=7

    # Hydro values
    hyd_hdg=np.arctan2(filters.lowpass_fir(hyd_v,winsize),
                       filters.lowpass_fir(hyd_u,winsize))[1:-1]

    # Swim-based values:
    x_lp=filters.lowpass_fir(swim_x,winsize)
    y_lp=filters.lowpass_fir(swim_y,winsize)
    u=np.diff(x_lp)/np.diff(swim_t)
    v=np.diff(y_lp)/np.diff(swim_t)
    hdg=np.arctan2(v,u)
    swim_turn=(np.diff(hdg) -np.pi)%(2*np.pi) + np.pi
    swim_spd=np.sqrt(u**2+v**2)
    swim_spd_ctr=0.5*(spd[1:]+spd[:-1])
    swim_hdg_ctr=0.5*(hdg[1:]+hdg[:-1]) # this is already relative to the flow
    
    # Geo-based values
    geo_x_lp=filters.lowpass_fir(geo_x,winsize)
    geo_y_lp=filters.lowpass_fir(geo_y,winsize)
    u=np.diff(geo_x_lp)/np.diff(swim_t)
    v=np.diff(geo_y_lp)/np.diff(swim_t)
    geo_hdg=np.arctan2(v,u)
    geo_turn=(np.diff(geo_hdg) -np.pi)%(2*np.pi) + np.pi
    geo_spd=np.sqrt(u**2+v**2)
    geo_spd_ctr=0.5*(spd[1:]+spd[:-1])
    geo_hdg_ctr=0.5*(geo_hdg[1:]+geo_hdg[:-1])

    rel_hdg_ctr=geo_hdg_ctr - hyd_hdg

    stride=1
    data['u']=swim_spd_ctr[::stride]
    #data['v']=swim_turn[::stride]
    data['v']=geo_turn[::stride]
    data['hdg']=(rel_hdg_ctr[::stride] +np.pi) %(2*np.pi) - np.pi
    data['N']=len(data['u'])
    
    data['swim_x']=x_lp[1:-1][::stride]
    data['swim_y']=y_lp[1:-1][::stride]
    data['geo_x']=geo_x[1:-1][::stride]
    data['geo_y']=geo_y[1:-1][::stride]
    data['t']=t[1:-1][::stride]
    
# So the usual moveHMM approach
# is using step length, and change in heading.
# that's probably a good place for me to start.

def load_model(model_file):
    model_pkl=model_file+'.pkl'
    if utils.is_stale(model_pkl,[model_file]):
        sm = pystan.StanModel(file=model_file)
        with open(model_pkl, 'wb') as fp:
            pickle.dump(sm, fp)
    else:
        with open(model_pkl,'rb') as fp:
            sm=pickle.load(fp)
    return sm

class show_model(object):
    cmap_state=cm.Dark2
    
    def __init__(self,sm,opt,data):
        self.nparam=len(self.params)
        self.sm=sm
        self.opt=opt
        self.data=data
        self.print_model()
        self.fig_summary()
    def print_model(self):
        self.print_parameters()
        self.print_transitions()
    def print_transitions(self):
        data=self.data ; opt=self.opt
        print()
        print("     To")    
        for j in range(data['K']):
            if j==0:
                print("From ",end='')
            else:
                print("     ",end='')

            for k in range(data['K']):
                print( "%.5f"%opt['theta'][j,k],end=' ')
            print()
    def print_parameters(self):
        data=self.data
        opt=self.opt
        for k in range(data['K']):
            freq= (opt['z_star']-1==k).sum() / len(opt['z_star'])
            param_str=self.parameter_string(k=k)
            
            print(f"State {k} {100*freq:4.1f}%: {param_str}")
            
class show_model_01(show_model):
    params=['Swim speed','Turn angle']

    def parameter_string(self,k):
        opt=self.opt
        data=self.data
        return f"speed ~ Gamma(μ={opt['spd_alpha'][k]/data['spd_beta'][k]:.3f})  turn ~ vM(conc={opt['lambda'][k]:5.2f}) "
    
    def fig_summary(self):
        opt=self.opt
        data=self.data
        
        fig=plt.figure(20)
        fig.clf()
        gs=gridspec.GridSpec(2*self.nparam,2)

        ax_geo=fig.add_subplot(gs[0:self.nparam,0])
        ax_swim=fig.add_subplot(gs[self.nparam:,0])

        for k in range(self.data['K']):
            color=self.cmap_state.colors[k]
            sel=(opt['z_star']-1==k)
            ax_geo.plot(  data['geo_x'][sel],   data['geo_y'][sel],  'o', color=color)
            ax_swim.plot( -data['swim_y'][sel], data['swim_x'][sel], 'o', color=color)
        
        ax_geo.axis('equal')
        ax_swim.axis('equal')

        for p in range(self.nparam):
            ax=fig.add_subplot(gs[2*p:2*(p+1),1])
            for k in range(data['K']):
                color=self.cmap_state.colors[k]
                self.plot_parameter(p=p,k=k,ax=ax,color=color)
            ax.legend(loc='upper right',title=self.params[p])
        self.fig=fig
                
    def plot_parameter(self,p,k,ax,color):
        if p==0:
            speeds=np.linspace(0,1.00,1000)
            # scipy takes the alpha shape parameter, but have to additionally scale it
            P_speed=stats.gamma(a=self.opt['spd_alpha'][k], scale=1./self.data['spd_beta'][k] )
            ax.plot( speeds, P_speed.pdf( speeds ),
                     label=f'State {k+1}',color=color)
            if k==self.data['K']+1:
                ax.axvline( P_speed.median() )
            
        elif p==1:
            turns=np.linspace(-np.pi,np.pi,200)
            ax.plot( turns, stats.vonmises.pdf( turns, loc=0, kappa=self.opt['lambda'][k]),
                     label=f'State {k+1}',color=color)



if 0:            
    sm=load_model('stan_hmm_v01.stan')
    opt=sm.optimizing(data=data)
    show_model_01(sm,opt,data)
    # When both states have the same std.dev. of swim speed, the model tends to
    # use only angle distribution (tau>0.25) or only speeds (tau<0.2) states.

class show_model_02(show_model_01):
    params=show_model_01.params + ['Geog. Heading']
    def parameter_string(self,k):
        opt=self.opt
        data=self.data
        s=super(show_model_02,self).parameter_string(k)
        return s + f" hdg ~ vM({opt['hdg_mean'][k]:5.2f},{opt['hdg_conc'][k]:5.2f}) "
    def plot_parameter(self,p,k,ax,color):
        if p<2:
            super(show_model_02,self).plot_parameter(p,k,ax,color)
        elif p==2:
            hdgs=np.linspace(-np.pi,np.pi,200)
            # Flip to geographic convention so -hdg is left, +hdg is right.
            ax.plot( -hdgs, stats.vonmises.pdf( hdgs, loc=self.opt['hdg_mean'][k],
                                               kappa=self.opt['hdg_conc'][k]),
                     label=f'State {k+1}',color=color)
        
# sm=load_model('stan_hmm_v02.stan')
sm=load_model('stan_hmm_v03.stan')
opt=sm.optimizing(data=data)
show_model_02(sm,opt,data)

##

# Effective sampling will require a stronger control
# on which states are which
samples=sm.sampling(data=data)

##


# Manually tweaking a gamma to match the global distribution
# of velocities suggests beta~25
plt.figure(10).clf()

plt.hist(data['u'],40,density=1)
x=np.linspace(0,1.25,200)
plt.plot( x, stats.gamma(8,scale=1./25).pdf(x) )
plt.plot( x, stats.norm(0.27,0.07).pdf(x) )

