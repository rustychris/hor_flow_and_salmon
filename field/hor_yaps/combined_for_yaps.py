"""
Read receiver data, split into manageable chunks (~6h), and check
for clock resets (which are convered to two independent stations)

Pass to R/yaps for fitting clock drift model.

This is adapted from split_for_yaps.py, but to write the whole period
out to a csv.
"""
import datetime
import os
import numpy as np
import xarray as xr
import re
from stompy import utils
import matplotlib.pyplot as plt
import prepare_pings
import parse_tek as pt
import pandas as pd
import six
##
utils.path("../circulation")

six.moves.reload_module(prepare_pings)

# ping_matcher_2018 may need some manual tweaks to deal with
# receivers that are too far out of sync.
pm=prepare_pings.ping_matcher_2018(split_on_clock_change=True)
pm.allow_dual_bad_pings=True

# Looks like it's probably okay
for d in pm.all_detects:
    print(f"{d.name.item()} {str(d.time.values[0])} -- {str(d.time.values[-1])}")
    

def fmt(d):
    return utils.to_datetime(d).strftime("%Y%m%dT%H%M")

##

# full range that seems to have a chance of resolving tracks
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

pm_nomp=pm.remove_multipath()

##
force=False
# these are short but not empty. explicitly ignore.
name_blacklist=['SM1.0','SM4.1', 'SM3.0', 'SM4.1']

for period in zip(breaks[:-1],breaks[1:]):
    period_dir=f"yaps/full/{fmt(period[0])}-{fmt(period[1])}"
    print(f"Processing {period_dir}")
    if not os.path.exists(period_dir):
        os.makedirs(period_dir)
        
    fn=os.path.join(period_dir,'all_detections.csv')
    if os.path.exists(fn) and not force:
        print(f"{fn} exists -- will skip")
        continue
    else:
        pm_clip=pm_nomp.clip_time(period)

    dfs=[]
    hydro_recs=[]

    for d in pm_clip.all_detects:
        if d.dims['index']==0: continue # clipped away
        if d.name.item() in name_blacklist:
            print(f"{d.name.item()} is in the blacklist -- ignore")
            continue
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

    hydros=pd.DataFrame(hydro_recs)
    hydros.to_csv(os.path.join(period_dir,'hydros.csv'),index=False)

##
import matplotlib.pyplot as plt 
from matplotlib import patches

# Plot the time ranges for each station
fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot()

stations=list(np.unique( [d.station.item() for d in pm_clip.all_detects] ))


for d in pm_clip.all_detects:
    if len(d.time)==0:
        continue
    station_i=stations.index(d.station.item())
    rect=patches.Rectangle( xy=[d.time.values[0],station_i-0.2],
                            width=d.time.values[-1]-d.time.values[0],
                            height=0.4,edgecolor='k',facecolor='0.6')
    ax.add_patch(rect)

    ax.text(d.time.values[0],station_i-0.15,d.name.item())

ax.axis(xmin=period[0],
        xmax=period[1],
        ymin=-0.5,ymax=12.5)

ax.yaxis.set_ticks(np.arange(len(stations)))
ax.yaxis.set_ticklabels(stations)
ax.xaxis_date()

# Can ignore a lot of those.
# Just need
# SM9.0, SM8.0, SM4.0, SM3.1, SM1.1, AM9.0
# AM8.8, AM5.0, AM4.0 AM3.3, AM2.0 AM1.0

# Middle of the period: SM1.2   break is at numpy.datetime64('2018-03-16T01:52:47')
#                       SM3.2   break is at numpy.datetime64('2018-03-21T00:03:38')

# Ignore SM4.1, SM3.0, SM1.0, SM4.1
