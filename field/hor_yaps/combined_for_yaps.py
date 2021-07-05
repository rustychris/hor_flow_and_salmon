"""
Read receiver data, split into manageable chunks (~6h), and check
for clock resets (which are convered to two independent stations)

Pass to R/yaps for fitting clock drift model.

This is adapted from split_for_yaps.py, but to write the whole period
out to a csv.
"""
import datetime
import os
from stompy import utils
utils.path("../tags")

import numpy as np
import xarray as xr
import re
import matplotlib.pyplot as plt

import prepare_pings
import parse_tek as pt
import pandas as pd
import six

import seawater

##

six.moves.reload_module(pt)
six.moves.reload_module(prepare_pings)

year=2020

if year==2018:
    # ping_matcher_2018 may need some manual tweaks to deal with
    # receivers that are too far out of sync.
    pm=prepare_pings.ping_matcher_2018(split_on_clock_change=True)
    pm.allow_dual_bad_pings=True
    # full range that seems to have a chance of resolving tracks
    # Should include the overall start and end times, too.
    breaks=[np.datetime64("2018-03-13 19:00:00"),
            np.datetime64('2018-03-16 01:52:47'), # Break for SM1.1 / SM1.2
            np.datetime64('2018-03-21 00:03:38'), # Break for SM3.1 / SM3.2
            # That made too long of a period -- sync model never finishes
            np.datetime64('2018-03-24 00:00:00'), # attempt more sync periods
            np.datetime64('2018-03-26 00:00:00'),
            np.datetime64('2018-04-01 00:00:00'),
            np.datetime64('2018-04-06 00:00:00'),
            np.datetime64('2018-04-10 00:00:00'),
            np.datetime64("2018-04-15 01:00:00")]
    # these are short but not empty. explicitly ignore.
    name_exclude=['SM1.0','SM4.1', 'SM3.0', 'SM4.1']
elif year==2020:
    pm=prepare_pings.ping_matcher_2020(split_on_clock_change=True)
    pm.allow_dual_bad_pings=True
    # full range that seems to have a chance of resolving tracks
    # This almost certainly too early and too late.
    breaks=[np.datetime64('2020-03-01T00:00:00'),
            # Make a nice manageable chunk in the middle for testing
            # But there were no fish in that chunk
            np.datetime64('2020-03-17T00:00:00'),
            np.datetime64('2020-03-17T06:00:00'),
            # So make a nice chunk that seems to have some fish:
            # Note that this slicing cannot at the moment be handled in yaps,
            # because there are clock resets that create multiple copies of
            # the same hydrophone, which angers yaps.
            np.datetime64('2020-04-09 11:00:00'),
            np.datetime64('2020-04-09 17:00:00'),
            # and generous end point
            np.datetime64('2020-05-20T00:00:00')]

    # SM3 has no valid data - prepare_pings comments very few
    # detections, no dbg file.
    min_duration=np.timedelta64(1,'h')

    nonshort=[]
    for df in pm.all_detects:
        t_min=max(breaks[0],df.time.values.min())
        t_max=min(breaks[-1],df.time.values.max())
        if t_max-t_min>min_duration:
            nonshort.append(df)
        else:
            print(f"{df.name.item()} is too short: {(t_max-t_min)/np.timedelta64(1,'h'):.2f}h")
    print(f"{len(nonshort)} of {len(pm.all_detects)} hydrophone records were long enough")
    
    pm.all_detects=nonshort
    name_exclude=[]
## 
# Looks like it's probably okay
for d in pm.all_detects:
    print(f"{d.name.item()} {str(d.time.values[0])} -- {str(d.time.values[-1])}")

# Check for unique beacon ids:
beacon_to_station={}
for d in pm.all_detects:
    station=d.station.item()
    tag=d.beacon_id.item()
    if tag in beacon_to_station:
        assert beacon_to_station[tag]==station
    else:
        beacon_to_station[tag]=station

##     
def fmt(d):
    return utils.to_datetime(d).strftime("%Y%m%dT%H%M")

#--
pm_nomp=pm.remove_multipath()
#-- 

total_detections=sum( [det.dims['index'] for det in pm.all_detects])
total_nomp_detections=sum( [det.dims['index'] for det in pm_nomp.all_detects])

print("Total detections (via prepare_pings): ", total_detections)
# Total detections (via prepare_pings):  5244204
print("Total detections after multipath filter (via prepare_pings): ", total_nomp_detections)
# Total detections after multipath filter (via prepare_pings):  4338914

##-- 
force=True
count_in=0
count_out=0

breaks=pm_nomp.infer_breaks(t_start=np.datetime64('2020-03-01T00:00:00'),
                            t_stop =np.datetime64('2020-05-20T00:00:00'))

for period in zip(breaks[:-1],breaks[1:]):
    period_dir=f"yaps/full/{year}/{fmt(period[0])}-{fmt(period[1])}"
    print(f"Processing {period_dir}")
    if not os.path.exists(period_dir):
        os.makedirs(period_dir)
        
    fn=os.path.join(period_dir,'all_detections.csv')
    if os.path.exists(fn) and not force:
        print(f"{fn} exists -- will skip")
        continue
    else:
        pm_clip=pm_nomp.clip_time(period)

    pm_clip.drop_by_name(name_exclude)
        
    period_detections=sum( [det.dims['index'] for det in pm_clip.all_detects])
    print(f"  detections {period_detections}")
    count_in+=period_detections
        
    dfs=[]
    hydro_recs=[]
    
    for d in pm_clip.all_detects:
        df1=d.to_dataframe()

        df2=df1.reset_index().loc[:, ['tag']]
        df2['serial']=d.name.item()
        df2['epo']=df1['epoch'].values
        df2['frac']=df1.usec.values / 1.0e6
        dfs.append(df2)

        rec=dict(serial=d.name.item(),
                 x=d.rx_x.item(),
                 y=d.rx_y.item(),
                 z=d.rx_z.item(),
                 sync_tag=d.beacon_id.item(),
                 idx=1+len(hydro_recs))
        hydro_recs.append(rec)
        
    df_all=pd.concat(dfs)
    df_all.to_csv(os.path.join(period_dir,'all_detections.csv'),index=False)
    
    print(f"  all_detections {len(df_all)}")
    count_out+=len(df_all)

    # Make sure sync_tags are unique:
    # 2020, SM10.0 and SM11.1 both give FF14 as their tag
    hydros=pd.DataFrame(hydro_recs)
    hydros.to_csv(os.path.join(period_dir,'hydros.csv'),index=False)

    if 0:
        fig=pm_clip.plot_chunk_stations()
        fig.axes[0].set_title(str(period))
        fig.savefig(os.path.join(period_dir,'chunk_stations.png'),dpi=200)

    if 1: # Write temperature data, too.
        dfs_temp=[det[ ['temp','epoch'] ].to_dataframe()
                  for det in pm_clip.all_detects]

        df_temp=pd.concat(dfs_temp).sort_values('epoch')
        df_temp['temp']=df_temp['temp'].astype(np.float64) # some dataframes have this as object

        # bin average at 15 minutes.
        bin_size=15*60
        df_bin_temp=df_temp.groupby(df_temp['epoch']//bin_size).mean()

        # T90 conversion pointless since temp sensor is only down to 0.1degC,
        # salinity, temperature, pressure
        df_bin_temp['ss']=seawater.svel(0.0,df_bin_temp['temp'].values,0.0)

        df_bin_temp=df_bin_temp.rename({'epoch':'ts'},axis=1)
        df_bin_temp[ ['ts','ss'] ].to_csv(os.path.join(period_dir,'soundspeed.csv'),index=False)

    

# Total count unclipped is about 5.2M, but I think that includes times
# when receivers weren't in the water, or maybe very few were in
# the water.
print(f"Clipping {count_in} total in, {count_out} total out")
# Clipping 2765802 total in, 2765610 total out

##

# Chunks:
#   Three options
#  (1) Adjust tags to reflect which receivers were active.
#      i.e. if FF14 got moved midway through, create a fake
#      tag FF140 for the second period, and change all rx 
#      of FF14 after the change to be FF140.

#  (2) Split on every time there was a possible move in the
#      hydrophone.
#      Downside is that location solutions for other hydrophones
#      will be on shorter datasets, so probably more noise.
#      This is probably the best first step, though.

#  (3) Assume hydrophones didn't move too much and didn't change
#      clock too much, and just merge.
## 


# How many stations, and for how much of the overall period,
# do we have temperature?


plt.figure(10).clf()

for det in pm_nomp.all_detects:
    sel=np.isfinite(det.temp.values)
    t=utils.unix_to_dt64(det.epoch.values)
    plt.plot( t[sel], det.temp.values[sel])

# temp seems good!
# O(1 degC) variation across hydrophones at a single time.

# So how does yaps want SS estimates?
# ss_data is a table with ts timestamp and ss sound speed in m/s
# So I have to average over the receivers and write out to csv.

##

