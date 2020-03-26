"""
Read receiver data, split into manageable chunks (~6h), and check
for clock resets (which are convered to two independent stations)

Pass to R/yaps for fitting clock drift model.
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


# Write out some 3 hour chunks
# Try a 6 hour chunk
# period=[np.datetime64("2018-03-18 00:00:00"),
#         np.datetime64("2018-03-18 06:00:00")]

pm.max_shift=60
pm.max_drift=0.0 # any faster?
pm_nomp=pm.remove_multipath()

# run 4 hours chunks with a 1 hr pad before and after
# These are too broad
# t_min_global=utils.floor_dt64( min( [d.time.values[0] for d in pm.all_detects] ),
#                                np.timedelta64(1,'D') )
# t_max_global=utils.ceil_dt64( max( [d.time.values[-1] for d in pm.all_detects]),
#                               np.timedelta64(1,'D') )
t_min_global=np.datetime64("2018-03-12 00:00:00")
t_max_global=np.datetime64("2018-04-15 00:00:00")
pad=np.timedelta64(1,'h')
duration=np.timedelta64(4,'h')

periods=[]

t_start=t_min_global
while t_start<t_max_global:
    t_stop=t_start+duration
    periods.append([t_start-pad,t_stop+pad])
    t_start+=duration


# add_beacon_list is called on a ds_total with index=0, and no rx.
# This is failing - maybe because there aren't enough beacon receptions??
# Okay - if nobody gets merged, then there isn't an rx dimension.
# First go around, only 4 rx have pings.  and they are only
# hearing themselves.  Either (a) match_all_by_similarity needs to
# merge them anyway (and later split_for_yaps code will toss the whole
# dataset), or it should return None, and downstream code deals with that.

##

# Testing with re-inclusion of AM3.  This period has at least
# one track that would really benefit from AM3
force=False
omit_am3=False
#periods=[ [np.datetime64('2018-03-20T07:00:00'),
#           np.datetime64('2018-03-20T13:00:00')] ]

for period in periods:
    period_dir=f"yaps/{fmt(period[0])}-{fmt(period[1])}"
    if not os.path.exists(period_dir):
        os.makedirs(period_dir)
        
    fn=os.path.join(period_dir,'all_detections.csv')
    if os.path.exists(fn) and not force:
        print(f"{fn} exists -- will skip")
        continue

    # Keep the python-side pre-match code
    # It's slow, but allows for greater clock shifts than yaps alone, and
    # also provides a chance to clean out some data that causes yaps
    # issues (like beacon tags with insufficient receptions)

    pm_clip=pm_nomp.clip_time(period)
    # temporary
    pm_clip.allow_dual_bad_pings=True
    # /temporary
    ds_total=pm_clip.match_all_by_similarity()
    if ds_total is None:
        print(f"Processing for {period_dir} aborted -- insufficient beacon receptions")
        continue
    pings_orig=ds_total

    # This was necessary in a test, and in this 2018-03-18 period, also
    # necessary, to avoid a cryptic error about subscript bounds
    if omit_am3:
        rx_mask=[not r.item().startswith('AM3') for r in pings_orig.rx]
        print("Still removing AM3, which seems bad")
        pings=pings_orig.isel(rx=rx_mask)
    else:
        pings=pings_orig

    # remove clock offsets --
    mat=pings.matrix.values
    deltas=np.zeros(mat.shape[1],np.float64)
    for _ in range(5):
        mat_d = mat + deltas[None,:]
        new_deltas=np.nanmean( mat_d - np.nanmean(mat_d,axis=1)[:,None],axis=0)
        new_deltas=np.nan_to_num(new_deltas,0)

        print(f"new deltas: {np.mean(new_deltas**2)}")
        deltas-=new_deltas
    pings['matrix_orig']=pings['matrix']
    pings['matrix'].values[:,:]=mat+deltas[None,:]
    
    beacon_tags=pings.rx_beacon.values

    is_beacon=np.array( [ t in beacon_tags
                          for t in pings.tag.values ] )
    is_multirx=(np.isfinite(pings.matrix).sum(axis=1)>1).values
    is_triplerx=(np.isfinite(pings.matrix).sum(axis=1)>2).values

    # less restrictive than triplerx -- limit the above to tags
    # that at least once are seen by 3 rx
    good_tags=[t # .item()
               for t in np.unique(pings.isel(index=is_triplerx).tag.values)]
    is_goodtag=np.array( [ t in good_tags
                           for t in pings.tag.values])

    mask=is_goodtag & is_multirx

    bpings=pings.isel(index=mask)
    b0=bpings.to_dataframe()

    b1=b0[ ~b0.matrix.isnull() ].reset_index().loc[:, ['matrix','rx','tag']]
    b1['epo']=np.floor(b1.matrix)
    b1['frac']=b1.matrix - b1.epo
    b2=b1.rename(columns={'rx':'serial'})
    del b2['matrix']
    # everything but ts, which is just a text timestamp.
    b2.to_csv(fn,index=False)

    #-

    # generate the equivalent of ssu1$hydros
    # Try omitting sync_tag for sync_tags that are not heard by at least 3 rxs.
    hydros=xr.Dataset()
    hydros['serial']=pings.rx
    hydros['x']=pings.rx_x
    hydros['y']=pings.rx_y
    hydros['z']=0*pings.rx_z # this z info is probably useless.
    hydros['sync_tag']=pings.rx_beacon
    hydros['idx']=('rx',), 1+np.arange(pings.dims['rx'])
    df=hydros.to_dataframe()
    sync_tags=df['sync_tag'].values
    if isinstance(sync_tags[0],np.ndarray):
        # gets weird -- sometimes this is necessary
        df['sync_tag']=[t.item() for t in sync_tags]

    # good_tags=b2.tag.unique()
    # during testing, this kills FF24
    df.loc[~df['sync_tag'].isin(good_tags), 'sync_tag']=np.nan
    df.to_csv(os.path.join(period_dir,'hydros.csv'),index=False)


##

# AM3:
# periods=[ [np.datetime64('2018-03-20T07:00:00'),
#            np.datetime64('2018-03-20T13:00:00')] ]
# Aligning 1381 hits at AM3.3 with 1426 at SM9.0
# MATCH
# Found 1 potential sets of matches, longest is 1369 (i=0)

# That much seems fine.

# rerun all, including AM3...
