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

fig_dir="fig_analysis-20201015"+vel_suffix
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

## 
df_start=track_common.read_from_folder('screen_final')

##

df_start['track'].apply(track_common.calc_velocities,
                        model_u='model_u'+vel_suffix,
                        model_v='model_v'+vel_suffix)

# Do this after the swimspeeds are computed, otherwise when
# a track leaves and comes back, there will be a swim speed
# that jumps from the exit point to the re-entry point
track_common.clip_to_analysis_polygon(df_start,'track')

## 
# Day-night variability

t_min=df_start.track.apply( lambda t: t.tnum.min() )
t_max=df_start.track.apply( lambda t: t.tnum.min() )
t_mid=0.5*(t_min+t_max)
df_start['t_mid']=t_mid
tod_mid=(t_mid-7*3600)%86400. # PDT
df_start['tod_mid_s']=tod_mid

tod_mid_angle=df_start.tod_mid_s/86400 * 2*np.pi
df_start['tod_mid_angle_pdt']=tod_mid_angle

df_start['tod_mid_octant']=1.5 + 3*np.floor(df_start['tod_mid_s']/(3*3600.))

# Pattern of time-of-day and swim speed
df_start['mean_swim_urel']=df_start.track.apply( lambda t: np.nanmean( t.swim_urel) )
df_start['mean_swim_lateral']=df_start.track.apply( lambda t: np.nanmean( np.abs(t.swim_vrel)) )


df_start['mean_gnd_urel']=df_start.track.apply( lambda t: np.nanmean( t.ground_urel) )
df_start['mean_gnd_mag'] =df_start.track.apply( lambda t: np.nanmean( t.groundspeed) )

for track in df_start.track.values:
    track['model_mag']=np.sqrt(track['model_u'+vel_suffix]**2 +
                               track['model_v'+vel_suffix]**2)
    
df_start['mean_model_mag']=df_start.track.apply( lambda t: np.nanmean( t.model_mag) )

## 

# circular histogram

fig=plt.figure(30)
fig.clf()
ax=fig.add_subplot(111,projection='polar')
ax.set_rorigin(-15)
ax.set_theta_direction(-1) # make it clock-like
ax.set_theta_offset(np.pi/2)

ax.hist(tod_mid_angle,
        bins=np.linspace(0,2*np.pi,25))
ax.plot(tod_mid_angle,0*tod_mid_angle,'k.')
        
hours=np.arange(24) 
ax.set_xticks( hours/24 * 2*np.pi)
ax.set_xticklabels( ["%d"%h for h in hours] )
ax.grid(0,axis='y')
plt.setp(ax.get_yticklabels(),visible=0)

# circular mean
circ_mean=stats.circmean(tod_mid_angle)
ax.plot( [circ_mean],[-10],'ro')
circ_std=stats.circstd(tod_mid_angle) # 2.642

thetas=np.linspace(circ_mean-circ_std,
                   circ_mean+circ_std,
                   100)
ax.plot( thetas, -10*np.ones_like(thetas),'r-')

# 7:10am - 7:10pm is good approx. for sunrise/sunset
# PDT
theta_night=2*np.pi/24 * np.linspace(19.17,24+7.17,50)
theta_day  =2*np.pi/24 * np.linspace(7.17,19.17,50)
ax.plot( theta_night, -8*np.ones_like(theta_night),'k-',lw=3,solid_capstyle='butt')
ax.plot( theta_day,   -8*np.ones_like(theta_night),'y-',lw=3,solid_capstyle='butt')

fig.savefig(os.path.join(fig_dir,'diel-presence.png'))

##

# same but simpler histplot

fig=plt.figure(31)
fig.clf()
ax=fig.add_subplot(111)

distro=df_start.groupby( np.floor(df_start['tod_mid_s']/3600.).astype(np.int32) ).size()
sns.barplot(distro.index,distro.values, ax=ax)
ax.set_ylabel('Tracks/hour')
ax.set_xlabel('Hour of day (PDT)')
ax.spines['top'].set_visible(0)
ax.spines['right'].set_visible(0)

# This is a rough approach to a circular kde
vm=stats.vonmises
theta=np.linspace(0,2*np.pi,100)
hour=24*theta/(2*np.pi)
kappa=2.5 # eh?  Like a bandwidth parameter for kde

dens=np.zeros_like(theta)
for track_theta in df_start.tod_mid_angle_pdt:
    dens+= vm.pdf(kappa=kappa,x=theta,loc=track_theta)
dens /= len(df_start)
# np.trapz(dens,theta) ==> 1.0  It's a proper distribution

# Scale to match the height of the histogram
# i.e. go from prob density of 1/rad to tracks hour
# the -0.5 is to match the stagger of the bars --
# the interval [0,1) is a bar centered on 0.
ax.plot(hour-0.5,dens * len(df_start)*(2*np.pi)/24,
        color='0.5',
        label='$\kappa$=%g'%kappa)

ax.plot(df_start.tod_mid_angle_pdt * 24/(2*np.pi),
        0*df_start.tod_mid_angle_pdt,'k+')
    
fig.savefig(os.path.join(fig_dir,'diel-presence-bar.png'))

## 

def boxplot_n(x,y,data,**kw):
    ax=sns.boxplot(x=x,y=y,data=data,**kw)
    # show N for each:
    for grp_i,(xval,grp) in enumerate(data.groupby(x)[y]):
        ax.text( grp_i, grp.median(), "%d"%(grp.count()), ha='center',fontsize=12,
                 color='w',va='center')
##    
fig=plt.figure(32)
fig.clf()
ax=fig.add_subplot(111)

# bin by 3 hour chunks
boxplot_n(x='tod_mid_octant',y='mean_swim_urel',data=df_start,ax=ax)

ax.set_xlabel('Middle of 3-hour bin (h)')
ax.set_ylabel('Mean downstream swimming')

fig.savefig(os.path.join(fig_dir,'octant-swim_urel.png'))

## 

fig=plt.figure(33)
fig.clf()
ax=fig.add_subplot(111)

# bin by 3 hour chunks
df_start['tod_mid_octant']=1.5 + 3*np.floor(df_start['tod_mid_s']/(3*3600.))
boxplot_n(x='tod_mid_octant',y='mean_swim_lateral',data=df_start,ax=ax)
ax.set_xlabel('Middle of 3-hour bin (h)')
ax.set_ylabel('Mean lateral swimming magnitude')
fig.savefig(os.path.join(fig_dir,'octant-swim_lateral.png'))

##

# Stats on octant / lateral
tod_mid_rad=df_start['tod_mid_s'] * 2*np.pi / 86400
from astropy.stats import rayleightest

# 0.94.
p_rayleigh_arrival=rayleightest(tod_mid_rad)
print("Rayleigh statistic for time of day of arrival: %.4f"%p_rayleigh_arrival)

# hmm - so how to translate this to testing whether u_lateral has a diel
# component?
# Can I just use u_lateral as the weighting? No, the result is not independent
# of the mean value.
# 0.1715
lateral=df_start['mean_swim_lateral'].values

## What if I brute force it?

lat_vec=np.c_[ np.cos(tod_mid_rad)*lateral,
               np.sin(tod_mid_rad)*lateral ]

lat_vec_mean=lat_vec.mean(axis=0)
lat_vec_mag=utils.mag(lat_vec_mean)


shuffle_mags=[]
shuffle_vecs=[]
N=10000 # should be ~1e2 bigger 
for _ in range(N):
    shuffle_rad=np.random.permutation(tod_mid_rad)
    shuffle_mean=np.mean( np.c_[ np.cos(shuffle_rad)*lateral,
                                 np.sin(shuffle_rad)*lateral ],
                          axis=0)
    shuffle_vecs.append(shuffle_mean)
    shuffle_mag=utils.mag(shuffle_mean)
    shuffle_mags.append(shuffle_mag)

shuffle_vecs=np.array(shuffle_vecs)

## 
plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
ax.plot(lat_vec[:,0],lat_vec[:,1],'g+')
ax.axhline(0,color='k',lw=0.5)
ax.axvline(0,color='k',lw=0.5)

ax.plot([lat_vec_mean[0]],[lat_vec_mean[1]],'go',zorder=3)
stride=slice(None,None,N//1000)
ax.plot(shuffle_vecs[stride,0],shuffle_vecs[stride,1],'ko',alpha=0.1)

mag_sort=np.sort(shuffle_mags)

p=1-np.searchsorted(mag_sort,lat_vec_mag)/float(1+len(mag_sort))
if mag_sort[-1]<lat_vec_mag:
    print("p < %.6f"%p)
else:
    print("p ~ %.6f"%p)

ax.axis('equal')
ax.set_adjustable('box')
ax.axis(xmin=-0.15,xmax=0.15,ymin=-0.15,ymax=0.15)
fig.savefig(os.path.join(fig_dir,'diel-lateral-scatter.png'),dpi=200)

## 
# how does that compare to using a multilinear regression?
# About an order of magnitude difference in p, but
# whatever.
# higher order fourier components don't help. Not surprising.
# the box plot made it looks pretty sinusoidal.

for order in range(1,5):
    Xcols=[ np.ones( (len(tod_mid_rad),1) )]

    for o in range(order):
        Xcols.append( np.c_[ np.cos((1+o)*tod_mid_rad),
                             np.sin((1+o)*tod_mid_rad)] )
    X=np.concatenate(Xcols,axis=1)
    Y=lateral

    model=sm.OLS(Y,X).fit()

    print("Order %d, BIC %.3f"%(order,model.bic))

##

fig=plt.figure(34)
fig.clf()
ax=fig.add_subplot(111)

boxplot_n(x='tod_mid_octant',y='mean_gnd_urel',data=df_start,ax=ax)
ax.set_xlabel('Middle of 3-hour bin (h)')
ax.set_ylabel('Mean downstream groundspeed')
fig.savefig(os.path.join(fig_dir,'octant-gnd_urel.png'))

##

fig=plt.figure(35)
fig.clf()
ax=fig.add_subplot(111)

boxplot_n(x='tod_mid_octant',y='mean_gnd_mag',data=df_start,ax=ax)
ax.set_xlabel('Middle of 3-hour bin (h)')
ax.set_ylabel('Mean groundspeed')
fig.savefig(os.path.join(fig_dir,'octant-gnd_speed.png'))

##

fig=plt.figure(36)
fig.clf()
ax=fig.add_subplot(111)

boxplot_n(x='tod_mid_octant',y='mean_model_mag',data=df_start,ax=ax)
ax.set_xlabel('Middle of 3-hour bin (h)')
ax.set_ylabel('Mean water speed')
fig.savefig(os.path.join(fig_dir,'octant-water_speed.png'))

##
@memoize.memoize(lru=10)
def fetch_and_parse(local_file,url,**kwargs):
    if not os.path.exists(local_file):
        if not os.path.exists(os.path.dirname(local_file)):
            os.makedirs(os.path.dirname(local_file))
        utils.download_url(url,local_file,on_abort='remove')
    return pd.read_csv(local_file,**kwargs)

## 
# longer time scale trends

fig=plt.figure(37).clf()

fig,(ax,ax_lat,ax2,ax_dens)=plt.subplots(4,1,sharex=True,num=37)
fig.set_size_inches((9.5,9.0),forward=True)

times=utils.unix_to_dt64(df_start.t_mid.values)
ax.plot(times, df_start.mean_swim_urel, 'g.',label='Mean downstream swimming')
ax_lat.plot(times, df_start.mean_swim_lateral, 'b.',label='Mean lateral swimming')
ax.legend(loc='upper right')
ax_lat.legend(loc='upper right')
fig.autofmt_xdate()
ax.axhline(0.0,color='0.6',lw=1.0)

# MSD flows
msd_flow=fetch_and_parse(local_file="env_data/msd/flow-2018.csv",
                         url="http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/docs/B95820Q/2018/FLOW_15-MINUTE_DATA_DATA.CSV",
                         skiprows=3,parse_dates=['time'],names=['time','flow_cfs','quality','notes'])

ax2.plot(msd_flow.time,msd_flow.flow_cfs,label='MSD Flow (cfs)')
ax.axis(xmin=times.min(), xmax=times.max())

msd_turb=cdec.cdec_dataset('MSD',times.min(),times.max(),sensor=27,
                           cache_dir='env_data')
msd_turb['turb_lp']=('time',),signal.medfilt(msd_turb.sensor0027.values,5)

msd_tempF=cdec.cdec_dataset('MSD',times.min(),times.max(),sensor=25,
                            cache_dir='env_data')

ax_turb=ax2.twinx()
ax_turb.plot(msd_turb.time,msd_turb.sensor0027,label='MSD Turb. (NTU)',color='orange',alpha=0.2,lw=0.4)
ax_turb.plot(msd_turb.time,msd_turb.turb_lp,label='MSD Turb. (NTU)',color='orange')
ax2.legend(handles=ax2.lines+ax_turb.lines)
plt.setp(ax_turb.get_yticklabels(),color='orange')
plt.setp(ax2.get_yticklabels(), color=ax2.lines[0].get_color())

t=np.linspace(df_start.t_mid.min(),df_start.t_mid.max(),400)

for bw in [None,0.05]:
    kernel = stats.gaussian_kde(df_start.t_mid,bw_method=bw)
    dens=kernel(t)
    ax_dens.plot(utils.unix_to_dt64(t),kernel(t),label='KDE (bw=%s)'%bw)

ax_dens.legend(loc='upper right')
plt.setp(ax_dens.get_yticklabels(),visible=0)

fig.tight_layout()

# Nothing jumps out, except timing 
fig.savefig(os.path.join(fig_dir,'timeseries-swim_flow_turb.png'))

##

# Time of arrival vs. flow conditions
fig=plt.figure(38).clf()

fig,(ax_dens,ax2,ax_temp)=plt.subplots(3,1,sharex=True,num=38)
fig.set_size_inches((9.5,9.0),forward=True)

times=utils.unix_to_dt64(df_start.t_mid.values)
t=np.linspace(df_start.t_mid.min(),df_start.t_mid.max(),400)

for bw in [None,0.05]:
    kernel = stats.gaussian_kde(df_start.t_mid,bw_method=bw)
    dens=kernel(t)
    ax_dens.plot(utils.unix_to_dt64(t),kernel(t),label='KDE (bw=%s)'%bw)

ax_dens.plot(times,0*df_start.t_mid,'k+',label='Tag arrivals')
    
ax_dens.legend(loc='upper left')
plt.setp(ax_dens.get_yticklabels(),visible=0)

# ax.plot(times, df_start.mean_swim_urel, 'g.',label='Mean downstream swimming')
# ax_lat.plot(times, df_start.mean_swim_lateral, 'b.',label='Mean lateral swimming')
# ax.legend(loc='upper right')
# ax_lat.legend(loc='upper right')
# fig.autofmt_xdate()
# ax.axhline(0.0,color='0.6',lw=1.0)

## MSD flows
utils.path("../../model/suntans")
import common

dates=dict(start_date=np.datetime64("2018-03-01"),
           end_date=np.datetime64("2018-04-15"))

msd_flow=common.msd_flow(**dates)
msd_velo=common.msd_velocity(**dates)

# msd_flow=fetch_and_parse(local_file="env_data/msd/flow-2018.csv",
#                          url="http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/docs/B95820Q/2018/FLOW_15-MINUTE_DATA_DATA.CSV",
#                          skiprows=3,parse_dates=['time'],names=['time','flow_cfs','quality','notes'])

ax2.plot(msd_flow.time,msd_flow.flow_m3s,label='MSD Flow (m$^3$s)')
ax_dens.axis(xmin=times.min(), xmax=times.max())

msd_turb=cdec.cdec_dataset('MSD',times.min(),times.max(),sensor=27,
                           cache_dir='env_data')
msd_turb['turb_lp']=('time',),signal.medfilt(msd_turb.sensor0027.values,5)

msd_temp=cdec.cdec_dataset('MSD',times.min(),times.max(),sensor=25,
                           cache_dir='env_data')
msd_temp['temp']=('time'), (msd_tempF['sensor0025']-32)*5./9

ax_turb=ax2.twinx()
ax_turb.plot(msd_turb.time,msd_turb.sensor0027,label='MSD Turb. (NTU)',color='orange',alpha=0.2,lw=0.4)
ax_turb.plot(msd_turb.time,msd_turb.turb_lp,label='MSD Turb. (NTU)',color='orange')
ax2.legend(handles=ax2.lines+ax_turb.lines,loc='upper left')
plt.setp(ax_turb.get_yticklabels(),color='orange')
plt.setp(ax2.get_yticklabels(), color=ax2.lines[0].get_color())

ax_temp.plot(msd_temp.time,msd_temp.temp,label='MSD Temp ($^{\circ}$C)')
ax_temp.legend(loc='upper left')

fig.tight_layout()

fig.savefig(os.path.join(fig_dir,'timeseries-arrival_flow_turb.png'))

## 

# And presence, swimming spatially -- look at screen_track_with_hydro
# all_segs=

# Seem to have lost my turbidity plots...

# Add turbidity, flow, doy to df_start
df_start['turb']=np.interp( df_start.t_mid,
                            utils.to_unix(msd_turb.time.values),msd_turb['turb_lp'].values)
df_start['flow_m3s']=np.interp( df_start.t_mid,
                                utils.to_unix(msd_flow.time.values), msd_flow.flow_m3s)
df_start['velo_ms']=np.interp( df_start.t_mid,
                               utils.to_unix(msd_velo.time.values), msd_velo.velocity_ms)
df_start['time_mid']=utils.unix_to_dt64(df_start.t_mid)
df_start['doy']=df_start.time_mid.dt.dayofyear

##

turb_quant=30.0
df_start['turb_quant']=np.round(df_start.turb/turb_quant)*turb_quant

fig=plt.figure(40)
fig.set_size_inches( (4.5,4), forward=True)
fig.clf()
fig,axs=plt.subplots(2,1,num=40,sharex=True)

for ax,y in zip(axs,['mean_swim_urel','mean_swim_lateral']):
    boxplot_n(x='turb_quant',y=y,data=df_start,ax=ax)
    if y=='mean_swim_urel':
        ax.set_ylabel('$u_{rel}$ m s$^{-1}$')
    else:
        ax.set_ylabel('$|v_{rel}|$ m s$^{-1}$')

axs[0].set_xlabel("")
axs[1].set_xlabel('Turbidity (ntu), rounded')
            
fig.subplots_adjust(left=0.17,top=0.95,right=0.95)
fig.savefig(os.path.join(fig_dir,'turb-swimming.png'))

##

doy_quant=3.0
df_start['doy_quant']=np.round(df_start.doy/doy_quant)*doy_quant

fig=plt.figure(41)
fig.set_size_inches( (4.5,4), forward=True)
fig.clf()
fig,axs=plt.subplots(2,1,num=41,sharex=True)

for ax,y in zip(axs,['mean_swim_urel','mean_swim_lateral']):
    boxplot_n(x='doy_quant',y=y,data=df_start,ax=ax)
    if y=='mean_swim_urel':
        ax.set_ylabel('$u_{rel}$ m s$^{-1}$')
    else:
        ax.set_ylabel('$|v_{rel}|$ m s$^{-1}$')

axs[0].set_xlabel("")
axs[1].set_xlabel('Day of year (rounded)')
            
fig.subplots_adjust(left=0.17,top=0.95,right=0.95)
fig.savefig(os.path.join(fig_dir,'doy-swimming.png'))

##

flow_quant=40.0
df_start['flow_quant']=np.round(df_start.flow_m3s/flow_quant)*flow_quant

fig=plt.figure(42)
fig.set_size_inches( (4.5,4), forward=True)
fig.clf()
fig,axs=plt.subplots(2,1,num=42,sharex=True)

for ax,y in zip(axs,['mean_swim_urel','mean_swim_lateral']):
    boxplot_n(x='flow_quant',y=y,data=df_start,ax=ax)
    if y=='mean_swim_urel':
        ax.set_ylabel('$u_{rel}$ m s$^{-1}$')
    else:
        ax.set_ylabel('$|v_{rel}|$ m s$^{-1}$')

axs[0].set_xlabel("")
axs[1].set_xlabel('Flow (m s$^{-1}$), rounded')
            
fig.subplots_adjust(left=0.17,top=0.95,right=0.95,bottom=0.16)
fig.savefig(os.path.join(fig_dir,'flow-swimming.png'))

##

# Try fitting linear model
df_to_fit=df_start.loc[:, ['mean_swim_urel','mean_swim_lateral','turb','flow_m3s','velo_ms','doy'] ]

print(f"N = {len(df_start)}")

# Dropped the pandas corr() calls in favor of scipy.stats
from scipy.stats import spearmanr,kendalltau

for endog in ['mean_swim_urel','mean_swim_lateral']:
    for exog in ['turb','flow_m3s','doy','velo_ms']:
        s_corr,s_p=spearmanr(a=df_start[endog],
                             b=df_start[exog])
        print("%20s ~ %10s  Spearman rho=%.3f  p-value=%.6f"%(endog, exog, s_corr,s_p))
        # k_corr,k_p=kendalltau(x=df_start[endog],
        #                       y=df_start[exog])
        # print("%20s ~ %10s  Kendall corr=%.3f  p-value=%.6f"%("", "", k_corr,k_p))
    print()

# This used to yield the best correlation with flow, then doy, and turb trailing
# But with the updated (cfg10->cfg12) hydro, and clipping tracks above the junction,
# it's now a tossup.
#  N = 121
#        mean_swim_urel ~       turb  Spearman rho=-0.319  p-value=0.000368
#        mean_swim_urel ~   flow_m3s  Spearman rho=-0.315  p-value=0.000430
#        mean_swim_urel ~        doy  Spearman rho=-0.338  p-value=0.000149
#        mean_swim_urel ~    velo_ms  Spearman rho=-0.300  p-value=0.000844
#  
#     mean_swim_lateral ~       turb  Spearman rho=0.000  p-value=0.999353
#     mean_swim_lateral ~   flow_m3s  Spearman rho=-0.053  p-value=0.560852
#     mean_swim_lateral ~        doy  Spearman rho=-0.047  p-value=0.612347
#     mean_swim_lateral ~    velo_ms  Spearman rho=-0.048  p-value=0.603427

##
def stand(x):
    return (x-x.min()) / (x.max()-x.min())

plt.figure(50).clf()
plt.plot( df_start.time_mid, stand(df_start.turb), '.', color='tab:brown',label='Turbidity')
plt.plot( df_start.time_mid, stand(df_start.flow_m3s), '.', color='tab:blue',label='Flow')
plt.plot( df_start.time_mid, stand(df_start.velo_ms), '.', color='tab:green',label='Velo')
plt.plot( df_start.time_mid, stand(df_start.doy), '.', color='tab:red',label='Day of year')

plt.plot( df_start.time_mid, 1-stand(df_start.mean_swim_urel), 'o', color='k')
plt.legend()

##

import statsmodels.formula.api as smf

# ordered from best BIC (most negative) to worst (least negative)
for formula in ['mean_swim_urel ~ flow_m3s',
                'mean_swim_urel ~ turb',
                'mean_swim_urel ~ doy',
                'mean_swim_urel ~ velo_ms',
                'mean_swim_urel ~ flow_m3s + turb',
                'mean_swim_urel ~ flow_m3s + doy',
                'mean_swim_urel ~ turb + doy',
                'mean_swim_urel ~ flow_m3s + turb + doy']:
    mod = smf.ols(formula=formula, data=df_start)
    res = mod.fit()

    print(f"{formula:40}: R^2={res.rsquared:.3f}   AIC={res.aic:.2f}   BIC={res.bic:.2f}")
                
## 
model = smf.ols(formula='mean_swim_urel ~ flow_m3s', data=df_start).fit()

# R-squared: 0.086
print(model.summary())

## 
# Of the simple linear models, this is the best
# Scatter plot it

plt.figure(42).clf()
fig,ax=plt.subplots(1,1,num=42)

ax.plot( df_start['flow_m3s'], df_start['mean_swim_urel'], 'k.')
pred=model.predict(df_start)
order=np.argsort(df_start['flow_m3s'].values)
ax.plot( df_start['flow_m3s'].values[order], pred[order], 'g-')

ax.set_xlabel('Flow (m$^3$ s$^{-1}$)')
ax.set_ylabel('$u_{rel}$ (m s$^{-1}$)')
fig.savefig(os.path.join(fig_dir,'scatter-flow-u_rel.png'),dpi=200)

##

# Fits at the sample level, across all tracks, to compare local hydro speed
# and river flow.

for idx,row in df_start.iterrows():
    row['track']['id'] = idx
    row['track']['flow_m3s']=row['flow_m3s']
    row['track']['velo_ms']=row['velo_ms']
    row['track']['turb']=row['turb']
    
seg_tracks=pd.concat( [ track.iloc[:-1,:]
                        for track in df_start['track'].values ] )

seg_tracks['hydro_speed']=np.sqrt( seg_tracks['model_u'+vel_suffix].values**2 +
                                   seg_tracks['model_v'+vel_suffix].values**2 )
seg_tracks['swim_lat']=np.abs(seg_tracks['swim_vrel'])

## 
# Global relationship between swim speed and hydro speed
num=51
plt.figure(num).clf()
fig,(ax_hydro, ax_flow)=plt.subplots(1,2,num=num)
fig.set_size_inches((6,2.75),forward=True)

endog=seg_tracks['swim_urel'].values

for ax,exog,exog_label in [
        (ax_hydro,seg_tracks['hydro_speed'].values,'Hydro speed (m/s)'),
        (ax_flow, seg_tracks['flow_m3s'].values,   'Flow (m3/s)') ]:
    ax.plot( exog, endog, '.', color='tab:blue', ms=1,alpha=0.2, zorder=-1)
    order=np.argsort(exog)
    N=401

    # No weighting in this version -- see plot_tracks.
    exog_med=medfilt( exog[order], N)
    endog_med=medfilt( endog[order], N)

    ax.plot( exog_med[N:-N], endog_med[N:-N], '-',color='k',zorder=1)

    ax.set_xlabel(exog_label)
    
    ax.axhline(0,color='k',lw=1.2,zorder=0)
    ax.axis(ymin=-0.8,ymax=0.8)

ax_hydro.axis(xmin=0,xmax=0.8)
ax_flow.axis(xmin=40,xmax=250)
ax_hydro.set_ylabel(r'$lon$ (m/s)')
plt.setp(ax_flow.get_yticklabels(), visible=0)
fig.subplots_adjust(left=0.15,right=0.92,top=0.97,bottom=0.18,wspace=0.09)

fig.savefig(os.path.join(fig_dir,'swim_urel-vs-hydro_and_flow.png'),dpi=200)

##     
# Pearson and Spearman tests:
df_to_fit=seg_tracks.loc[:, ['swim_urel','swim_lat','hydro_speed','flow_m3s','velo_ms','turb'] ]

print(f"N={len(df_to_fit)}")
selector=['swim_urel','swim_lat'], ['hydro_speed','flow_m3s','velo_ms','turb']
# Straight linear correlation:
print("Pearson, linear-----------")
print(df_to_fit.corr().loc[selector])
print()
print("Spearman, rank------------")
print(df_to_fit.corr(method='spearman').loc[selector])
#print(df_to_fit.corr(method='kendall'))
print()
print("--------------------------")

## linear models

# What about decomposing velocity using river flow?
mb=np.polyfit( seg_tracks.velo_ms.values, seg_tracks.hydro_speed.values,1)

vel_anom=seg_tracks.hydro_speed.values - np.polyval(mb,seg_tracks.velo_ms.values)

plt.figure(50).clf()
fig,ax=plt.subplots(num=50)

ax.plot( seg_tracks.tnum, seg_tracks.velo_ms,'g.',label='River velocity (m/s)')
ax.plot( seg_tracks.tnum, seg_tracks.hydro_speed,'b.',label='Local velocity (m/s)')
ax.plot( seg_tracks.tnum, vel_anom, '.',color='orange',label='Local velocity anom.')

seg_tracks['hydro_anom']=vel_anom
##

import statsmodels.formula.api as smf

seg_tracks['flow_100m3s']=seg_tracks['flow_m3s']/100.

results=[]
for formula in [ 'swim_urel ~ hydro_speed',
                 'swim_urel ~ velo_ms',
                 'swim_urel ~ turb',
                 'swim_urel ~ hydro_speed + turb',                 
                 'swim_urel ~ hydro_speed + velo_ms',
                 'swim_urel ~ turb + velo_ms',
                 'swim_urel ~ hydro_speed + turb + velo_ms',
                 # 'swim_urel ~ flow_100m3s',
                 #'swim_urel ~ hydro_anom',
                 # 'swim_urel ~ hydro_anom + velo_ms',
                 #'swim_urel ~ hydro_speed + flow_100m3s'
]:
    mod = smf.ols(formula=formula, data=seg_tracks)
    res = mod.fit()
    results.append(res)

order=np.argsort( [res.bic for res in results] )
recs=[] # assemble a results dataframe while we're at it.
for idx in order:
    res=results[idx]
    print("-"*50)
    print("Formula: ",res.model.formula)
    print(f"  R^2: {res.rsquared:.4f}   AIC: {res.aic:.1f} BIC: {res.bic:.1f}  f-pvalue {res.f_pvalue:.2e} n={res.nobs:g}")
    params=res.params
    conf_ints=res.conf_int()
    print("  Parameters: ")
    for pnum,pname in enumerate(res.model.exog_names):
        print(f"    {pname:15}  {params[pname]:.4f}   [{conf_ints.loc[pname,0]:.4f}   {conf_ints.loc[pname,1]:.4f}]")
    print()

    recs.append( dict( idx=idx, formula=res.model.formula,
                      rsq=res.rsquared, aic=res.aic, bic=res.bic,
                      p_value=res.f_pvalue, n=res.nobs ) )

##

fits=pd.DataFrame(recs)

ffits=fits.loc[:,['formula','rsq','bic']].copy()
ffits['rsq']=ffits['rsq'].apply(lambda x: "%.4f"%x)
ffits['bic']=ffits['bic'].apply(lambda x: "%.0f"%x)
ffits['formula']=ffits['formula'].apply(lambda x: x.replace('swim_urel ~ ',''))

ffits.rename({'formula':'Model',
              'rsq':'R^2',
              'bic':'BIC'},
             inplace=True,axis=1)
print(ffits)
#print( fits.style.format({'rsq': '{:.4f}', 'bic': '{:.0f}'}) )
