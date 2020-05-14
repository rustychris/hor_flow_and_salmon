import os
import numpy as np
from stompy import utils
import pandas as pd

from local_config import cache_dir

pad=np.timedelta64(1,'D')

def get_wdl(local_file,url):
    local_path=os.path.join(cache_dir,local_file)
    utils.download_url(local_file=local_path,url=url)
    flow=pd.read_csv(local_path,skiprows=3,parse_dates=['time_pst'],
                     names=['time_pst','flow_cfs','quality','notes'])
    flow['flow_m3s']=msd_flow.flow_cfs*(0.3048)**3
    flow['time']=msd_flow['time_pst'] + np.timedelta64(8,'h')
    return flow

def msd_flow(start_date,end_date):
    flow=get_wdl(local_file="msd-flow-2018.csv",
                 url=("http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/"
                      "docs/B95820Q/2018/FLOW_15-MINUTE_DATA_DATA.CSV") )
    # Eventually handle other years.
    assert start_date-pad > flow.time.values[0]
    assert end_date_pad < flow.time.values[-1]
    return flow

def sjd_flow(start_date,end_date):
    flow=get_wdl(local_file="sjd-flow-2018.csv",
                 url=("http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/"
                      "docs/B95760/2018/FLOW_15-MINUTE_DATA_DATA.CSV") )
    assert start_date-pad > flow.time.values[0]
    assert end_date_pad < flow.time.values[-1]
    return flow

def oh1_flow(start_date,end_date):
    flow=get_wdl(local_file="oh1-flow-2018.csv",
                 url=("http://wdl.water.ca.gov/waterdatalibrary/docs/Hydstra/"
                      "docs/B95400Q/2018/FLOW_15-MINUTE_DATA_DATA.CSV") )
    assert start_date-pad > flow.time.values[0]
    assert end_date_pad < flow.time.values[-1]
    return flow
