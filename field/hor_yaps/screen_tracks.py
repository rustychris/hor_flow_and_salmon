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
    
    fig.savefig('merge-distribution-stddev.png')
##

# Write these out so we can get hydro
six.moves.reload_module(track_common)
track_common.dump_to_folder(df_merged,'merged_v00')

##

# Read them back in with hydro
df_mergedh=track_common.read_from_folder('mergedhydro_v00')
df_mergedh.rename( columns={'withhydro':'track'}, inplace=True)

# This is a beacon tag that was ignored for some periods
df_mergedh=df_mergedh.drop('FF01')

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
    
df['trimmed']=df['track'].apply(trim_weak_solutions)
# And limit to tracks that still exist
df_trimmed=df[ df['trimmed'].notnull() ]

print("%d tracks remain after trimming weak solutions and ends"%len(df_trimmed)) # 179

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

df_trimmed['trimmed'].apply(calc_groundspeed)
    
if 1: # Plot the global distribution of groundspeed
    all_spd=np.concatenate( [ df.groundspeed.values
                              for df in df_trimmed['trimmed'] ])
    all_spd=all_spd[np.isfinite(all_spd)]
    plt.figure(2).clf()
    fig,ax=plt.subplots(num=2)
    ax.hist(all_spd,bins=np.linspace(0,3.0,100),log=True)
    ax.set_ylabel('Segment count')
    ax.set_xlabel('Groundspeed (m s$^{-1}$)')
    fig.savefig('trimmed-distribution-groundspeed.png')

# Groundspeed is not a good metric for good/bad samples.
# Almost everything is below 1.0 m/s.

##     

#  Combine to a main csv and per-track csv that can be shipped over to cws-linuxmodeling
#  and I can add hydro velocity.

track_common.dump_to_folder(df_trimmed,'cleaned_v00')

##
df=df_trimmed.copy()

# A few more track parameters:
df['num_positions']=[ len(track) for track in df['trimmed'].values ]
df['duration_s']=[ (track.tnum.values[-1] - track.tnum.values[0])
                   for track in df['trimmed'].values ]

if 1:
    plt.figure(3).clf()
    fig,ax=plt.subplots(num=3)
    bins=np.logspace(-0.1,4,20)
    ax.hist(df.num_positions,bins=bins)
    ax.set_xscale('log')
    ax.set_xlabel('Count of positions in track')
    ax.set_ylabel('Occurrences')
    fig.savefig('track-position-count-hist.png')

    plt.figure(4).clf()
    fig,ax=plt.subplots(num=4)
    bins=np.logspace(-0.1,4,20)
    ax.hist(df.duration_s/60.,bins=bins)
    ax.set_xscale('log')
    ax.set_xlabel('Duration of track (min)')
    ax.set_ylabel('Occurrences')
    fig.savefig('track-duration-hist.png')

##

# Based on histograms, choose
good_duration=df['duration_s'] < 60*60
good_posn_count=df['num_positions']>=10

df_screen=df[good_duration & good_posn_count].copy()

# 129
print("After removing long-duration or small position count tracks %d left"%(len(df_screen)))

##

# Add velocities
df_screen['trimmed'].apply(track_common.calc_velocities,
                           model_u='model_u_surf',model_v='model_v_surf')

##

def add_seg_time_end(track):
    track['tnum_end'] = np.r_[ track['tnum'].values[1:], np.nan]
    track['x_end']= np.r_[ track['x'].values[1:], np.nan]
    track['y_end']= np.r_[ track['y'].values[1:], np.nan]
    
df_screen['trimmed'].apply(add_seg_time_end)

## 

track_common.dump_to_folder(df_screen,'screened')
