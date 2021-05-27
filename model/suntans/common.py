import os
import xarray as xr

import numpy as np
from stompy import utils
import pandas as pd

from local_config import cache_dir

pad=np.timedelta64(1,'D')

# Applied to WDL timestamps to get UTC
wdl_raw_to_utc=lambda t: t+np.timedelta64(8,'h')
max_gap_allowed_s=2.5*3600

def get_wdl(start_date,end_date,local_file,url):
    local_path=os.path.join(cache_dir,local_file)
    utils.download_url(local_file=local_path,url=url)
    
    data=pd.read_csv(local_path,skiprows=3,parse_dates=['time_pst'],
                     names=['time_pst','value','quality','notes'])
    data['time']=wdl_raw_to_utc(data['time_pst'])

    # Eventually handle other years.
    assert start_date-pad > data.time.values[0]
    assert end_date+pad < data.time.values[-1]
    sel=(data.time.values>=start_date-pad)&(data.time.values<end_date+pad)
    sel=sel&np.isfinite(data.value.values)
    data=data[sel]

    max_gap=np.diff(data.time.values).max() / np.timedelta64(1,'s')
    assert max_gap<max_gap_allowed_s,"Max gap %s > allowable %s"%(max_gap, max_gap_allowed_s)
    
    data_ds=xr.Dataset.from_dataframe(data.set_index('time'))
    
    assert np.all(np.isfinite(data_ds.value.values))
    
    return data_ds

def get_wdl_flow(*a,**kw):
    data_ds=get_wdl(*a,**kw)
    data_ds['flow_m3s']=data_ds['value']*(0.3048)**3
    return data_ds

def get_wdl_stage(*a,**kw):
    data_ds=get_wdl(*a,**kw)
    data_ds['stage_m']=data_ds['value']*0.3048
    return data_ds

def get_wdl_velocity(*a,**kw):
    data_ds=get_wdl(*a,**kw)
    data_ds['velocity_ms']=data_ds['value']*0.3048
    return data_ds

def msd_flow(start_date,end_date):
    flow=get_wdl_flow(start_date,end_date,
                      local_file="msd-flow-2018.csv",
                      url=("http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/"
                           "docs/B95820Q/2018/FLOW_15-MINUTE_DATA_DATA.CSV") )
    return flow

def msd_velocity(start_date,end_date):
    return get_wdl_velocity(start_date,end_date,
                            local_file="msd-velocity-2018.csv",
                            url=("https://wdlstorageaccount.blob.core.windows.net/continuousdata/"
                                 "docs/B95820Q/2018/VELOCITY_DATA.CSV"))

def sjd_flow(start_date,end_date):
    flow=get_wdl_flow(start_date,end_date,
                      local_file="sjd-flow-2018.csv",
                      url=("http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/"
                           "docs/B95760/2018/FLOW_15-MINUTE_DATA_DATA.CSV") )
    return flow

def oh1_flow(start_date,end_date):
    flow=get_wdl_flow(start_date,end_date,
                      local_file="oh1-flow-2018.csv",
                      url=("http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/"
                           "docs/B95400Q/2018/FLOW_15-MINUTE_DATA_DATA.CSV") )
    return flow

def oh1_stage(start_date,end_date):
    flow=get_wdl_stage(start_date,end_date,
                       local_file="oh1-stage-2018.csv",
                       url=("http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/"
                            "docs/B95400/2018/STAGE_15-MINUTE_DATA_DATA.CSV") )
    return flow

# BROKEN
def sjd_stage(start_date,end_date):
    flow=get_wdl_stage(start_date,end_date,
                      local_file="sjd-stage-2018.csv",
                      url=("http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/"
                           "docs/B95760/2018/STAGE_15-MINUTE_DATA_DATA.CSV") )
    return flow

def msd_stage(start_date,end_date):
    flow=get_wdl_stage(start_date,end_date,
                      local_file="msd-stage-2018.csv",
                      url=("http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/"
                           "docs/B95820/2018/STAGE_15-MINUTE_DATA_DATA.CSV") )
    return flow
