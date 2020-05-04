import matplotlib.pyplot as plt
from scipy import stats
import os
import track_common
import seaborn as sns
from stompy import memoize
from stompy.io.local import cdec
from scipy.stats.kde import gaussian_kde
from scipy import signal 
##
fig_dir="fig_analysis-20200427"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

## 
df_start=track_common.read_from_folder('screen_final')

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
    track['model_mag']=np.sqrt(track.model_u_surf**2 + track.model_v_surf**2)
    
df_start['mean_model_mag']=df_start.track.apply( lambda t: np.nanmean( t.model_mag) )
## 

# spot check track 71A9, and the times work out. tod_mid is seconds
# into the day, UTC.

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

fig=plt.figure(32)
fig.clf()
ax=fig.add_subplot(111)

# bin by 3 hour chunks
sns.boxplot(x='tod_mid_octant',y='mean_swim_urel',data=df_start,ax=ax)
ax.set_xlabel('Middle of 3-hour bin (h)')
ax.set_ylabel('Mean downstream swimming')
fig.savefig(os.path.join(fig_dir,'octant-swim_urel.png'))

## 

fig=plt.figure(33)
fig.clf()
ax=fig.add_subplot(111)

# bin by 3 hour chunks
df_start['tod_mid_octant']=1.5 + 3*np.floor(df_start['tod_mid_s']/(3*3600.))
sns.boxplot(x='tod_mid_octant',y='mean_swim_lateral',data=df_start,ax=ax)
ax.set_xlabel('Middle of 3-hour bin (h)')
ax.set_ylabel('Mean lateral swimming magnitude')
fig.savefig(os.path.join(fig_dir,'octant-swim_lateral.png'))

##


fig=plt.figure(34)
fig.clf()
ax=fig.add_subplot(111)

sns.boxplot(x='tod_mid_octant',y='mean_gnd_urel',data=df_start,ax=ax)
ax.set_xlabel('Middle of 3-hour bin (h)')
ax.set_ylabel('Mean downstream groundspeed')
fig.savefig(os.path.join(fig_dir,'octant-gnd_urel.png'))

##

fig=plt.figure(35)
fig.clf()
ax=fig.add_subplot(111)

sns.boxplot(x='tod_mid_octant',y='mean_gnd_mag',data=df_start,ax=ax)
ax.set_xlabel('Middle of 3-hour bin (h)')
ax.set_ylabel('Mean groundspeed')
fig.savefig(os.path.join(fig_dir,'octant-gnd_speed.png'))

##


fig=plt.figure(36)
fig.clf()
ax=fig.add_subplot(111)

sns.boxplot(x='tod_mid_octant',y='mean_model_mag',data=df_start,ax=ax)
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

# Nothing jumps out, except timimig 
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

# MSD flows
msd_flow=fetch_and_parse(local_file="env_data/msd/flow-2018.csv",
                         url="http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/docs/B95820Q/2018/FLOW_15-MINUTE_DATA_DATA.CSV",
                         skiprows=3,parse_dates=['time'],names=['time','flow_cfs','quality','notes'])

ax2.plot(msd_flow.time,msd_flow.flow_cfs,label='MSD Flow (cfs)')
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
