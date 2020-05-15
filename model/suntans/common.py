import os
import xarray as xr

import numpy as np
from stompy import utils
import pandas as pd

from local_config import cache_dir

pad=np.timedelta64(1,'D')

# Applied to WDL timestamps to get UTC
wdl_raw_to_utc=lambda t: t+np.timedelta64(8,'h')

def get_wdl(start_date,end_date,local_file,url):
    local_path=os.path.join(cache_dir,local_file)
    utils.download_url(local_file=local_path,url=url)
    flow=pd.read_csv(local_path,skiprows=3,parse_dates=['time_pst'],
                     names=['time_pst','flow_cfs','quality','notes'])
    flow['flow_m3s']=flow.flow_cfs*(0.3048)**3
    flow['time']=wdl_raw_to_utc(flow['time_pst'])

    # Eventually handle other years.
    assert start_date-pad > flow.time.values[0]
    assert end_date+pad < flow.time.values[-1]
    sel=(flow.time.values>=start_date-pad)&(flow.time.values<end_date+pad)
    flow=flow[sel]
    
    flow_ds=xr.Dataset.from_dataframe(flow.set_index('time'))
    
    assert np.all(np.isfinite(flow_ds.flow_m3s.values))
    
    return flow_ds

def msd_flow(start_date,end_date):
    flow=get_wdl(start_date,end_date,
                 local_file="msd-flow-2018.csv",
                 url=("http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/"
                      "docs/B95820Q/2018/FLOW_15-MINUTE_DATA_DATA.CSV") )
    return flow

def sjd_flow(start_date,end_date):
    flow=get_wdl(start_date,end_date,
                 local_file="sjd-flow-2018.csv",
                 url=("http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/"
                      "docs/B95760/2018/FLOW_15-MINUTE_DATA_DATA.CSV") )
    return flow

def oh1_flow(start_date,end_date):
    flow=get_wdl(start_date,end_date,
                 local_file="oh1-flow-2018.csv",
                 url=("http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/"
                      "docs/B95400Q/2018/FLOW_15-MINUTE_DATA_DATA.CSV") )
    return flow
