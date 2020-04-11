import matplotlib.pyplot as plt
from scipy import stats
import os
import track_common
import seaborn as sns
##
fig_dir="fig_analysis-20200411"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

## 
df_start=track_common.read_from_folder('screen_final')

##

# Day-night variability

t_min=df_start.track.apply( lambda t: t.tnum.min() )
t_max=df_start.track.apply( lambda t: t.tnum.min() )
t_mid=0.5*(t_min+t_max)
tod_mid=(t_mid-7*3600)%86400. # PDT
df_start['tod_mid_s']=tod_mid

tod_mid_angle=df_start.tod_mid_s/86400 * 2*np.pi
df_start['tod_mid_angle_pdt']=tod_mid_angle

df_start['tod_mid_octant']=1.5 + 3*np.floor(df_start['tod_mid_s']/(3*3600.))

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

# Pattern of time-of-day and swim speed

df_start['mean_swim_urel']=df_start.track.apply( lambda t: np.nanmean( t.swim_urel) )
df_start['mean_swim_lateral']=df_start.track.apply( lambda t: np.nanmean( np.abs(t.swim_vrel)) )

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

df_start['mean_urel']=df_start.track.apply( lambda t: np.nanmean( t.urel) )

## 

fig=plt.figure(32)
fig.clf()
ax=fig.add_subplot(111)

# bin by 3 hour chunks
sns.boxplot(x='tod_mid_octant',y='mean_swim_urel',data=df_start,ax=ax)
ax.set_xlabel('Middle of 3-hour bin (h)')
ax.set_ylabel('Mean downstream swimming')
fig.savefig(os.path.join(fig_dir,'octant-swim_urel.png'))
