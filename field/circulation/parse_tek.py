import pandas as pd
import logging as log
import xarray as xr
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from stompy import utils


def parse_tek(det_fn,cf2_fn=None,name=None):
    if cf2_fn is None:
        fn=det_fn.replace('.DET','.CF2')
        if os.path.exists(fn):
            cf2_fn=fn

    df=pd.read_csv(det_fn,
                   names=['id','year','month','day','hour','minute','second','epoch','usec',
                          'tag','nbwQ','corrQ','num1','num2','one','pressure','temp'])

    if 0:
        # this way is slow, and I *think* yields PST times, where epoch is UTC.
        dates=[ datetime.datetime(year=rec['year'],
                                  month=rec['month'],
                                  day=rec['day'],
                                  hour=rec['hour'],
                                  minute=rec['minute'],
                                  second=rec['second'])
                for idx,rec in df.iterrows()]
        df['time'] = utils.to_dt64(np.array(dates)) + df['usec']*np.timedelta64(1,'us')
    else:
        # this is quite fast and should yield UTC.
        # the conversion in utils happens to be good down to microseconds, so we can
        # do this in one go
        df['time'] = utils.unix_to_dt64(df['epoch'] + df['usec']*1e-6)
        
    # clean up time:
    bad_time= (df.time<np.datetime64('2018-01-01'))|(df.time>np.datetime64('2022-01-01'))
    df2=df[~bad_time].copy()

    # clean up temperature:
    df2.loc[ df.temp<-5, 'temp'] =np.nan
    df2.loc[ df.temp>35, 'temp'] =np.nan

    # clean up pressure
    # this had been limited to 160e3, but
    # AM9 has a bad calibration (or maybe it's just really deep)
    df2.loc[ df2.pressure>225e3, 'pressure']=np.nan
    df2.loc[ df2.pressure<110e3, 'pressure']=np.nan

    # trim to first/last valid pressure
    valid_idx=np.nonzero( np.isfinite(df2.pressure.values) )[0]
    df3=df2.iloc[ valid_idx[0]:valid_idx[-1]+1, : ].copy()

    df3['tag']=[ s.strip() for s in df3.tag.values ]

    ds=xr.Dataset.from_dataframe(df3)
    ds['det_filename']=(),det_fn

    if name is not None:
        ds['name']=(),name

    if cf2_fn is not None:
        # SM2 isn't getting the right value here.
        # looks like it would be FF13, but it never
        # hears FF13.
        cf=pd.read_csv(cf2_fn,header=None)
        local_tag=cf.iloc[0,1].strip()
        ds['beacon_id']=local_tag
        ds['cf2_filename']=(),cf2_fn


    return ds

def remove_multipath(ds):
    # Sometimes a single ping is heard twice (a local bounce)
    # when the same tag is heard in quick succession (<1s) drop the
    # second occurrence.
    # ds: Dataset filtered to a single receiver and a single beacon id.
    delta_us=np.diff(ds.time.values) / np.timedelta64(1,'us')

    if np.any(delta_us<=0):
        log.warning("%s has non-increasing timestamps (worst is %.2fs)"%(ds.det_filename.values,
                                                                         delta_us.min()*1e-6))
    # Warn and drop any non-monotonic detections.
    bounces=(delta_us<1e6) & (delta_us>0)
    valid=np.r_[ True, delta_us>1e6 ]
    return ds.isel(index=valid).copy()

