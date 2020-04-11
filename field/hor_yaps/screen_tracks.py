"""
Load the tracks, apply some trimming at the sample and track level, write out
in preparation for adding hydro data.
"""
import os
import six
import glob
import pandas as pd
import numpy as np
import track_common

##

fig_dir="figs20200406"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

##     
raw_csv_patt='yaps/full/v06/track*.csv'
raw_csvs=glob.glob(raw_csv_patt)

##
# 1. Read them all in

df=pd.DataFrame(dict(raw_fn=raw_csvs))

def ingest_csv(fn):
    csv_df=pd.read_csv(fn)
    del csv_df['Unnamed: 0']
    return csv_df

df['track']=df['raw_fn'].apply(ingest_csv)
df['basename']=df['raw_fn'].apply(os.path.basename)
df['id']=df['basename'].apply( lambda s: s.replace('track-','').replace('.csv','') )
del df['raw_fn']

##
# 2.  Merge any tracks which span multiple analysis periods

def merge(to_merge):
    if len(to_merge)==1:
        result=to_merge
    else:
        # When does each one start:
        raws=to_merge['track'].values
        tnum_starts=[df.tnum.min() for df in raws]
        order=np.argsort(tnum_starts)

        merged=raws[order[0]]
        for i in order[1:]:
            raw=raws[i]
            sel_monotonic= (raw.tnum.values>merged.tnum.values[-1])
            # just the ones that come after what we have already
            raw=raw.iloc[sel_monotonic,:]
            merged=pd.concat([merged,raw])

        result=to_merge.iloc[:1]
        result['track'] = [merged]

    result=result.reset_index()
    return result

df_merged=df.groupby('id',as_index=True).apply(merge)
df_merged=df_merged.set_index( df_merged.index.droplevel(1))

print("%d tracks remain after merging"%len(df_merged)) # 195

##

if 1: # Plot the global distribution of position standard deviations
    all_var_xy=np.concatenate( [ (df.x_sd**2 + df.y_sd**2).values
                                 for df in df_merged['track'] ])
    all_std_xy=np.sqrt(all_var_xy)
    # 61308 total position estimates

    # A lot of the variance there is driven by solutions with <3 receivers
    rx3_var_xy=np.concatenate( [ (df.x_sd**2 + df.y_sd**2).values[ df.num_rx.values>2 ]
                                 for df in df_merged['track'] ])
    rx3_std_xy=np.sqrt(rx3_var_xy)

    rx2_var_xy=np.concatenate( [ (df.x_sd**2 + df.y_sd**2).values[ df.num_rx.values>1 ]
                                 for df in df_merged['track'] ])
    rx2_std_xy=np.sqrt(rx2_var_xy)

    plt.figure(1).clf()
    fig,axs=plt.subplots(3,1,sharex=True,num=1)

    bins = np.logspace(-2,3,100)
    axs[0].set_xscale('log')
    axs[0].hist(all_std_xy,bins=bins) 
    axs[1].hist(rx2_std_xy,bins=bins) 
    axs[2].hist(rx3_std_xy,bins=bins)

    for ax,label in zip( axs, ['All','>=2 receivers','>=3 receivers']):
        ax.text(0.95,0.95,label,transform=ax.transAxes,ha='right',va='top')

    axs[2].set_xlabel('Standard deviation of position estimates')
    axs[0].set_ylabel('Count of position estimates')
    axs[1].set_ylabel('Count of position estimates')
    axs[2].set_ylabel('Count of position estimates')
    
    fig.savefig(os.path.join(fig_dir,'merge-distribution-stddev.png'))
##

# Write these out so we can get hydro
six.moves.reload_module(track_common)
track_common.dump_to_folder(df_merged,'merged_v00')

##

# Read them back in with hydro
df_mergedh=track_common.read_from_folder('mergedhydro_v00_sun')
if 'withhydro' in df_mergedh.columns:
    del df_mergedh['track']
    df_mergedh.rename( columns={'withhydro':'track'}, inplace=True)

##

df=df_mergedh

# Apply some prefiltering

# This got rid of some actually pretty nice tracks lik 7A96.
# Relax the sd test to 10.0m
    
# Toss solutions with <2 receivers and with std-dev greater than 10.0m
# Then trim the ends to start/end with a 3-rx solution.
def trim_weak_solutions(track):
    good_num_rx=track['num_rx']>= 2
    good_std = np.sqrt( (track.x_sd**2 + track.y_sd**2).values ) <= 10.0
    track=track[ good_num_rx & good_std ]
    
    triple_rx=np.nonzero((track['num_rx']>2).values)[0]
    if len(triple_rx)==0:
        return None
    else:
        return track.iloc[triple_rx[0]:triple_rx[-1]+1,:].copy()
    
df['track']=df['track'].map(trim_weak_solutions)
# And limit to tracks that still exist
df_trimmed=df[ df['track'].notnull() ]

print("%d tracks remain after trimming weak solutions and ends"%len(df_trimmed)) # 181

##

# Calculate duration of track, RMS velocity.
# assuming that the tracks are m

# For segment values, leave the *last* value nan.

def calc_groundspeed(track):
    dx=np.diff(track.x.values)
    dy=np.diff(track.y.values)
    dt=np.diff(track.tnum)
    speed=np.sqrt(dx**2+dy**2)/dt
    track['groundspeed']=np.r_[speed,np.nan]

df_trimmed['track'].apply(calc_groundspeed)
    
if 1: # Plot the global distribution of groundspeed
    all_spd=np.concatenate( [ df.groundspeed.values
                              for df in df_trimmed['track'] ])
    all_spd=all_spd[np.isfinite(all_spd)]
    plt.figure(2).clf()
    fig,ax=plt.subplots(num=2)
    ax.hist(all_spd,bins=np.linspace(0,3.0,100),log=True)
    ax.set_ylabel('Segment count')
    ax.set_xlabel('Groundspeed (m s$^{-1}$)')
    fig.savefig(os.path.join(fig_dir,'trimmed-distribution-groundspeed.png'))

# Groundspeed is not a good metric for good/bad samples.
# Almost everything is below 1.0 m/s.

##     

df=df_trimmed.copy()

# A few more track parameters:
df['num_positions']=[ len(track) for track in df['track'].values ]
df['duration_s']=[ (track.tnum.values[-1] - track.tnum.values[0])
                   for track in df['track'].values ]

if 1:
    plt.figure(3).clf()
    fig,ax=plt.subplots(num=3)
    bins=np.logspace(-0.1,4,20)
    ax.hist(df.num_positions,bins=bins)
    ax.set_xscale('log')
    ax.set_xlabel('Count of positions in track')
    ax.set_ylabel('Occurrences')
    fig.savefig(os.path.join(fig_dir,'track-position-count-hist.png'))

    plt.figure(4).clf()
    fig,ax=plt.subplots(num=4)
    bins=np.logspace(-0.1,4,20)
    ax.hist(df.duration_s/60.,bins=bins)
    ax.set_xscale('log')
    ax.set_xlabel('Duration of track (min)')
    ax.set_ylabel('Occurrences')
    fig.savefig(os.path.join(fig_dir,'track-duration-hist.png'))

##

# Based on histograms, choose
good_duration=df['duration_s'] < 60*60
good_posn_count=df['num_positions']>=10

df_screen=df[good_duration & good_posn_count].copy()

# 134
print("After removing long-duration or small position count tracks %d left"%(len(df_screen)))

##

# Add velocities
df_screen['track'].apply(track_common.calc_velocities,
                         model_u='model_u_surf',model_v='model_v_surf')

##

def add_seg_time_end(track):
    track['tnum_end'] = np.r_[ track['tnum'].values[1:], np.nan]
    track['x_end']= np.r_[ track['x'].values[1:], np.nan]
    track['y_end']= np.r_[ track['y'].values[1:], np.nan]
    
df_screen['track'].apply(add_seg_time_end)

## 

track_common.dump_to_folder(df_screen,'screened_sun')

track_common.dump_to_shp(df_screen,'screened_sun/tracks.shp',overwrite=True)

##

df=df_screen
gates=wkb2shp.shp2geom('track_gates-v00.shp')

recs=[]

for idx,row in df_screen.iterrows():
    track=row['track']
    cross=track_common.gate_crossings(track,gates)

    rec={'index':idx}
    for gidx,name in enumerate(gates['name']):
        if 'levee' in name: continue
        if 'point' in name: continue # never happens
        t_first=t_last=np.nan
        if len(cross)>0: # may have no crossings at all
            l2r_cross=np.nonzero( (cross['gate']==gidx)& cross['l2r'] )[0]
            if len(l2r_cross):
                t_first=cross['tnum'][ l2r_cross[0]  ]
                t_last =cross['tnum'][ l2r_cross[-1] ]
        rec[name+"_first"]=t_first
        rec[name+"_last"] =t_last
        rec[name+"_uncrossed"]= ((cross['gate']==gidx)&(~cross['l2r'])).sum()
    recs.append(rec)
df_crossings=pd.DataFrame(recs).set_index('index')

## 
# Add those into df:
df_with_crossings=pd.merge(df_screen,df_crossings,left_index=True,right_index=True)

## 

# Print some summary stats:
print(f"Count of tracks: {len(df_with_crossings)}")
for gidx,name in enumerate(gates['name']):
    if ('levee' in name) or (name=='point'): continue
    t_first=df_with_crossings[name+"_first"]
    t_last=df_with_crossings[name+"_last"]
    n_crossed=t_first.count()
    n_multiple=(t_first<t_last).sum()
    n_uncross=(df_with_crossings[name+"_uncrossed"]>0).sum()
    print(f"  {name:12}: {n_crossed:3} tracks, {n_multiple:2} recross, {n_uncross} uncross")

##

track_subs=[]
for idx,row in df_with_crossings.iterrows():
    t_min=np.nanmax( [row['top_of_array_last'],
                      row['track']['tnum'].min()] )
    # Ah - sj_lower_first is negative??
    t_max=np.nanmin( [row['sj_lower_first'],
                      row['hor_lower_first'],
                      row['track']['tnum'].max()])
    sel=( (t_min<=row['track']['tnum'].values)
          & (row['track']['tnum'].values<=t_max ) )
    track_sub=row['track'].iloc[sel,:].copy()
    track_subs.append(track_sub)

df_trim_crossings=df_with_crossings.copy()
df_trim_crossings['track']=track_subs
valid=np.array( [len(t)>0 for t in track_subs] )
df_trim_crossings=df_trim_crossings.iloc[valid,:].copy()

# 134 to 134 tracks
print("Trimming to top_of_array -- lower gates: %d to %d tracks"%
      (len(df_with_crossings),len(df_trim_crossings)))

# Clean up some unused columns
for col in ['id.1','index']:
    if col in df_trim_crossings:
        df_trim_crossings.drop(col,inplace=True,axis=1)

##

# Good time to think about predators
# Can I establish a set of parameters that, either individually or together,
# cluster predators and non-predators?

df=df_trim_crossings

for idx,row in df.iterrows():
    track=row['track']
    segs=track.iloc[:-1,:]
    
    # This is a bit noisy, though.
    # what is the expected value of actual swim spd?
    # see noise_study.py for some discussion of this,
    # but no solution.
    swim_spd=np.sqrt((segs['swim_u']**2 + segs['swim_v']**2))
    dt=(segs['tnum_end']-segs['tnum']).values
    assert np.all(dt>0)
    assert np.all(np.isfinite(swim_spd*dt))
    assert dt.sum()>0
    
    swim_spd_mean=(swim_spd*dt).sum() / dt.sum()
    df.loc[idx,'swim_spd_mean']=swim_spd_mean

    swim_spd_var=(swim_spd-swim_spd_mean)**2
    df.loc[idx,'swim_spd_std']=np.sqrt( (swim_spd_var*dt).sum() / dt.sum() )
    
##

plt.figure(14).clf()
fig,ax=plt.subplots(num=14)
ax.hist(df.swim_spd_mean,bins=25)
ax.set_xlabel('Mean swimming speed per track (m/s)')
fig.savefig('mean_swim_speed_per_track-v00.png')

##
plt.figure(15).clf()
fig,ax=plt.subplots(num=15)

sns.kdeplot(df.swim_spd_mean,df.swim_spd_std,
            shade_lowest=True,
            shade=True,
            #linewidths=0.5,
            levels=5,
            ax=ax,zorder=1)
ax.plot(df.swim_spd_mean, df.swim_spd_std, 'k.',ms=3,zorder=3)

ax.set_xlabel('Mean swimming speed per track (m/s)')
ax.set_ylabel('Std. dev. of swimming speed per track (m/s)')
fig.savefig('stddev_swim_speed_per_track-v00.png')
##

plt.figure(16).clf()
fig,ax=plt.subplots(num=16)

sns.kdeplot(df.swim_spd_mean,df.duration_s/60.,
            shade_lowest=True,
            shade=True,
            levels=5,
            ax=ax,zorder=1)
ax.plot(df.swim_spd_mean, df.duration_s/60., 'k.',ms=3,zorder=3)
ax.axis(ymin=0,xmin=0)

ax.set_xlabel('Mean swimming speed per track (m/s)')
ax.set_ylabel('Duration of track (min)')
fig.savefig('swim_speed_duration_per_track-v00.png')

## 

tagged_fish=pd.read_excel('../circulation/2018_Data/2018FriantTaggingCombined.xlsx')
real_fish_tags=tagged_fish.TagID_Hex
non_fish_tracks=~df.index.isin(real_fish_tags)

print("Potential non-fish tag ids:",
      " ".join(df.index.values[non_fish_tracks]) )

##

# Check for position error effect on mean speed.
df['rms_pos_error']=df.track.apply( lambda t: np.sqrt( (t.x_sd**2 + t.y_sd**2).mean() ) )

g=sns.jointplot(x=df.swim_spd_mean, y=df.rms_pos_error)
g.set_axis_labels('Mean swim speed per track (m/s)',
                  'RMS position error (m)')
g.fig.savefig('swim_speed_rms_pos_error-v00.png')

##

# What tags have the highest mean swim speeds?
fast_tags=np.argsort(df.swim_spd_mean.values)[-7:]
print(df.iloc[fast_tags,:].loc[:,'swim_spd_mean'])

##

df=df_trim_crossings

# Presumed predators:
non_smolt = ['7A51','746D','74B4']

smolt_sel = ~(df.index.isin(non_smolt))

df_smolt=df.loc[smolt_sel,:].copy()

print("Removed %d tracks as non-smolts, %d --> %d tracks"%
      (len(non_smolt), len(df), len(df_smolt)))

##

track_common.dump_to_folder(df_smolt, 'screen_final')
