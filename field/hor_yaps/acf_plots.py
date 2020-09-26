"""
Explore autocorrelation, partial autocorrelation, directional persistence
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
from stompy import nanpsd
from stompy import utils
from stompy.spatial import field
import seaborn as sns
import stompy.plot.cmap as scmap
turbo=scmap.load_gradient('turbo.cpt')

##
dem=field.GdalGrid("../../bathy/junction-composite-dem-no_adcp.tif")
clip=(647167.570005404,647416.10024743,
      4185000,4186500)
demcrop=dem.crop(clip)

##

def unwrap(angles):
    deltas=(np.diff(angles) + np.pi)%(2*np.pi) - np.pi
    return np.cumsum( np.r_[angles[0],deltas] )

def expand(data,time,dt_nom=5.0):
    # fill, based on nominal tag interval of 5 seconds
    di=np.round(np.diff(time)/dt_nom).astype(np.int32)
    assert di.min()>0
    n_expand = di.sum()+1
    tgts=np.r_[0,np.cumsum(di)]
    expanded=np.nan*np.zeros(n_expand)
    expanded[tgts]=data
    return expanded

##

input_path="screen_final"
#input_path="with_nonsmolt"
fig_dir=os.path.join(input_path,'figs-20200921')
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

df=track_common.read_from_folder(input_path)

# With multiple velocities, need this stanza to choose one

# RH: return to this code to compare heading distributions
vel_suffix='_top2m'
df['track'].apply(track_common.calc_velocities,
                  model_u='model_u'+vel_suffix,
                  model_v='model_v'+vel_suffix)

track_common.clip_to_analysis_polygon(df,'track')


## 
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
# dev for variable time

# Circular correlation
# https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Circular_Data_Correlation.pdf
# The correlation figure they give relies on a mean angle, in the same way that
# a linear correlation coefficient relies on a mean for each of the two populations.
# I guess what I want is to know how "similar" two angles are.
# So maybe what I want is the variance of the difference in heading

def autocorrcoef_circ(x,maxlags=None):
    # x: angles in radians
    N = len(x)
    if maxlags == None:
        maxlags = N - 1
    assert maxlags < N, 'lag must be less than len(x)'
    assert(np.isrealobj(x))
    
    r = np.zeros(1+maxlags, dtype=float)
    
    for k in range(0, maxlags+1):
        nk = N - k - 1
        a=x[0:nk+1]
        b=x[k:k+nk+1]
        valid=np.isfinite(a*b)
        d_angle=(a-b)[valid]

        # mean resultant length
        C1=np.cos(d_angle).sum()
        S1=np.sin(d_angle).sum()
        R1=np.sqrt(C1**2+S1**2)
        mR1=R1/len(d_angle)
        r[k]=mR1
    return r

def fig_turning(track,num=10,hdg_field='swim_hdg_rel'):
    dt_nom=5.0
    segs=track[:-1]
    nlag=min(len(segs)-1,200)

    hdg_unwrap=unwrap(segs[hdg_field].values)
    hdg_expand=expand(hdg_unwrap,segs.tnum_m,dt_nom=dt_nom)
    
    fig=plt.figure(num)
    fig.clf()
    gs=gridspec.GridSpec(2,3)
    ax= fig.add_subplot(gs[0,-1])
    ax_turn=fig.add_subplot(gs[1,-1])
    
    ax_geo=fig.add_subplot(gs[:,0])
    ax_swim=fig.add_subplot(gs[:,1])

    ax_geo.plot( track['x'], track['y'], 'k-',lw=0.5)
    # Marker every 2 minutes.
    mark_times=np.arange(track['tnum'].values[0],
                         track['tnum'].values[-1],
                         120)
    ax_geo.scatter( np.interp( mark_times, track['tnum'], track['x']),
                    np.interp( mark_times, track['tnum'], track['y']),
                    30, mark_times, cmap=turbo,alpha=0.75)
    ax_geo.xaxis.set_visible(0)
    ax_geo.yaxis.set_visible(0)
    ax_geo.axis('equal')
    
    demcrop.plot(ax=ax_geo,zorder=-5,cmap='gray',clim=[-20,10],interpolation='bilinear')
    demcrop.plot_hillshade(ax=ax_geo,z_factor=1,plot_args=dict(interpolation='bilinear',zorder=-4))
    pad=50
    ax_geo.axis( xmin=track['x'].min()-pad,
                 xmax=track['x'].max()+pad,
                 ymin=track['y'].min()-pad,
                 ymax=track['y'].max()+pad )

    ax_swim.plot( -track['swim_y'],track['swim_x'], 'k-',lw=0.5)
    ax_swim.scatter( -np.interp( mark_times, track['tnum'], track['swim_y']),
                     np.interp( mark_times, track['tnum'], track['swim_x']),
                     30, mark_times, cmap=turbo, alpha=0.75)
    ax_swim.xaxis.set_visible(0)
    ax_swim.yaxis.set_visible(0)
    ax_swim.axis('equal')

    common=dict(weight='bold',color='0.5',transform=ax_swim.transAxes,zorder=-1)
    ax_swim.text( 0.5, 0.99, 'Downstream',va='top',ha='center', **common)
    ax_swim.text( 0.5, 0.01, 'Upstream',  va='bottom',ha='center', **common)
    ax_swim.text( 0.02, 0.5, 'River\nLeft',va='center',ha='left',**common)
    ax_swim.text( 0.98, 0.5, 'River\nRight',  va='center',ha='right',**common)
    
    ace=nanpsd.autocorrcoef(hdg_expand,maxlags=nlag)
    ax.plot( dt_nom*np.arange(nlag+1), ace, 'go',ms=2,label='Autocorrcoef')
    
    ang_var=autocorrcoef_circ(hdg_expand,maxlags=nlag)
    ax.plot( dt_nom*np.arange(nlag+1), ang_var, 'o',color='orange',
             label='Circular. autocovariance',ms=2)
    ax.axhline(0.0,color='k',lw=0.5)
    ax.legend(loc='lower left',fontsize=8,bbox_to_anchor=(0,-0.36))

    # Autocorrelation of turn angles was basically noise
    turns=np.diff(hdg_expand)
    turns=turns[np.isfinite(turns)] * 180/np.pi
    ax_turn.hist(turns,bins=np.linspace(-180,180,50) )
    fig.subplots_adjust(top=0.95,bottom=0.05,left=0.01,right=0.99,
                        hspace=0.45)
    ax_turn.set_xticks([-90,0,90])
    ax_turn.set_xticklabels([r"-90$^{\circ}$","0",
                             r"90$^{\circ}$"])
    
    return fig

# Shows a looser turn angle distribution for the fish 7A96
# than the predator 74B4.

fig0=fig_turning(df.loc['7A96','track'],num=10)
fig0.axes[0].set_title('7A96')

fig1=fig_turning(df.loc['74B4','track'],num=11)
fig1.axes[0].set_title('74B4')

# For groundspeed heading (not relative to hydro),
# overall noisier for 74B4, not much change for 7A96.
fig0b=fig_turning(df.loc['7A96','track'],num=12,hdg_field='ground_hdg')
fig0b.axes[0].set_title('7A96')

fig1b=fig_turning(df.loc['74B4','track'],num=13,hdg_field='ground_hdg')
fig1b.axes[0].set_title('74B4')

##

for tag_i,tag in enumerate(df.index.values):
    fig=fig_turning(df.loc[tag,'track'])
    fig.axes[0].set_title(tag)
    fig.savefig(os.path.join(fig_dir,'acf_swim_hdg-%s.png'%tag))

##

# Similar, but show persistence of swimspeed, too
for tag_i,tag in enumerate(
        ['7A96'] #,'7A51','74B4','746D']
):
    track=df.loc[tag,'track']
    segs=track[:-1]

    plt.figure(60+tag_i).clf()
    fig,axs=plt.subplots(3,1,num=60+tag_i)
    fig.set_size_inches([9,7],forward=True)

    t_elapse=segs.tnum_m - segs.tnum[0]
    axs[0].plot(t_elapse, segs.swim_hdg_rel, label='Swim hdg (rel)')
    axs[0].plot(t_elapse, segs.swim_hdg_rel_uw, label='Swim hdg (rel) unwrap')
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('Track elapsed time (s)')

    ax_speed=axs[0].twinx()
    ax_speed.plot(t_elapse, segs.swim_speed,'g-')
    
    nlag=min(len(segs)//2-1,80)
    
    plot_pacf( segs['swim_hdg_rel_uw'], lags=nlag,ax=axs[1],method='ywmle')
    plot_pacf( segs['swim_speed'], lags=nlag,ax=axs[2],method='ywmle')

    axs[0].set_title(tag)
    axs[0].set_ylabel('Heading (rad)')
    ax_speed.set_ylabel('Swim speed (m/s)',color='g')
    plt.setp(ax_speed.get_yticklabels(),color='g')

    for label,ax in zip(['Swim heading','Swim speed'],axs[1:]):
        ax.set_title("PACF: "+label)
        ax.title.set_position([0.99,0.9])
        ax.title.set_ha('right')
        ax.title.set_va('top')

    axs[2].set_xlabel('Lag (samples)')
    plt.setp(axs[1].get_xticklabels(),visible=0)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir,'acf_swim_hdg_speed-%s.png'%tag))
    #break

##


# Across all tracks
# Still not worrying about proper circular stats or differing time steps.
from statsmodels.tsa.stattools import pacf_yw

recs=[]

for tag in df.index.values:
    track=df.loc[tag,'track']
    rec={}
    seg=track[:-1]
    nlag=min(len(seg)//2-1,20)
    pa_hdg = pacf_yw(seg['swim_hdg_rel_uw'],nlags=nlag,method='mle')
    pa_spd = pacf_yw(seg['swim_speed'],nlags=nlag,method='mle')

    for l in range(len(pa_hdg)):
        recs.append( dict(lag=l,tag=tag,
                          pacf_hdg=pa_hdg[l],
                          pacf_spd=pa_spd[l]) )
pacf_df=pd.DataFrame(recs)

plt.figure(70).clf()
fig,axs=plt.subplots(2,1,num=70)

sns.boxplot(x='lag',y='pacf_hdg',data=pacf_df,ax=axs[0])
sns.boxplot(x='lag',y='pacf_spd',data=pacf_df,ax=axs[1])

fig.savefig(os.path.join(fig_dir,'pacf_boxplot-hdg-speed.png'))

##

# What about those tracks where PACF(1)<0 ?

##

# Distribution of relative heading
for idx,row in df.iterrows():
    row['track']['id'] = idx

seg_tracks=pd.concat( [ track.iloc[:-1,:]
                        for track in df['track'].values ] )

# all_hdg_rel=np.concatenate( [ track.swim_hdg_rel.values for track in df['track'].values] )

# Weights -- segments per individual sum to unit weight
seg_counts=seg_tracks.groupby(seg_tracks.id).size()
weights=1./seg_counts[ seg_tracks.id]

##

plt.figure(10).clf()
fig,axs=plt.subplots(2,1,num=10)

axs[0].hist(seg_tracks.swim_hdg_rel.values,bins=100)

axs[1].hist(seg_tracks.swim_hdg_rel.values,weights=weights,bins=100)

# Pos vs. neg rheotaxis:
neg_sel=np.abs(seg_tracks.swim_hdg_rel.values)<np.pi/2
pos_sel=np.abs(seg_tracks.swim_hdg_rel.values)>np.pi/2

neg_count= weights[neg_sel].sum()
pos_count= weights[pos_sel].sum()
print(f"Negative rheotaxis: {100*neg_count/weights.sum():.1f}%") # 29.6%
print(f"Positive rheotaxis: {100*pos_count/weights.sum():.1f}%") # 70.4%

