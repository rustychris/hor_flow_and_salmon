"""
Read receiver data, split into manageable chunks (~6h), and check
for clock resets (which are convered to two independent stations)

Pass to R/yaps for fitting clock drift model.
"""
import datetime
import numpy as np
import re
from stompy import utils
import matplotlib.pyplot as plt
import prepare_pings
import parse_tek as pt
import pandas as pd
import six
## 
six.moves.reload_module(pt)
six.moves.reload_module(prepare_pings)
#--

# ping_matcher_2018 may need some manual tweaks to deal with
# receivers that are too far out of sync.
pm=prepare_pings.ping_matcher_2018(split_on_clock_change=True)

# Looks like it's probably okay
for d in pm.all_detects:
    print(f"{d.name.item()} {str(d.time.values[0])} -- {str(d.time.values[-1])}")
    
#--

# Write out some 3 hour chunks
period=[np.datetime64("2018-03-18 00:00:00"),
        np.datetime64("2018-03-18 03:00:00")]
def fmt(d):
    return utils.to_datetime(d).strftime("%Y%m%dT%H%M")
period_dir=f"yaps/{fmt(period[0])}-{fmt(period[1])}"
if not os.path.exists(period_dir):
    os.makedirs(period_dir)

# Keep the python-side pre-match code
# It's slow, but allows for greater clock shifts than yaps alone, and
# also provides a chance to clean out some data that causes yaps
# issues (like beacon tags with insufficient receptions)

# Super super slow when processing 6 hours at once
# And eventually gets stuck trying to line up AM8.8
pm.max_shift=60
pm.max_drift=0.0 # any faster?
pm_nomp=pm.remove_multipath()
pm_clip=pm_nomp.clip_time(period)
ds_total=pm_clip.match_all_by_similarity()
##

# HERE

# Not sure if it's better to write hydros.csv with all of the receivers,
# or a period-specific file

hydro_recs=[]

dfs=[] # Dataframes
for d in pm.all_detects:
    # Trim to this period
    t_sel=(d.time.values>=period[0])&(d.time.values<period[1])
    if not np.any(t_sel):
        continue
    
    trim_ds=d.isel(index=t_sel)
    # May also populate hydros here..

    df=pd.DataFrame()
    df['tag']=trim_ds['tag'].values
    df['epo']=trim_ds['epoch'].values
    df['frac']=trim_ds['usec'].astype(np.float64)/1.0e6
    df['serial']=trim_ds['name'].item()
    dfs.append(df)

    hydro_recs.append(dict(serial=trim_ds['name'].item(),
                           x=trim_ds['rx_x'].item(),
                           y=trim_ds['rx_y'].item(),
                           z=trim_ds['rx_z'].item(),
                           sync_tag=trim_ds['beacon_id'].item(),
                           idx=len(hydro_recs)+1))
    
# Merge those into a single dataframe, with columns
# serial (ie name), tag, epo, frac
df_period=pd.concat(dfs).sort_values(by=['epo','frac'])
df_period.to_csv(os.path.join(period_dir,'all_detections.csv'),index=False)

hydro_df=pd.DataFrame(hydro_recs)
hydro_df.to_csv(os.path.join(period_dir,'hydros.csv'),index=False)

# At this point, I think it's running into the same issues as before,
# where a sync tag that isn't at some point heard by self+2 rxs
# causes an issue.
    
    
