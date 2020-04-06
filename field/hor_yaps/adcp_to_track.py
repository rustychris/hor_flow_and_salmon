# For starters
import glob
import os
import xarray as xr
import pandas as pd

import track_common

##

adcp_fns=glob.glob("../adcp/040518_BT/*-avg_with_time.nc")

##

# Make sure -- these are probably local time
# yeah - times range from 8:50 to 14:15
# perfect for local, but if that were UTC,
# then depending on date it would be 7 or 8 hours off
# so that would be midnight to 7am.
# unlikely.

for fn in adcp_fns:
    ds=xr.open_dataset(fn)
    print(ds.time.values)
    ds.close()
##

# Study period all within daylight savings.  Will assume that ADCP
# was set to local = PDT = UTC-7
adcp_time_offset_to_utc=np.timedelta64(7,'h')

def adcp_to_track(adcp_fn):
    adcp=xr.open_dataset(adcp_fn)

    track=pd.DataFrame()
    track['x']=adcp['x_sample'].values
    track['y']=adcp['y_sample'].values
    track['d']=adcp['d_sample'].values
    # variables that can just be copied over:
    for v in ['z_bed','depth_bt','depth_m']:
        track[v]=adcp[v].values

    U_depth_avg=xr_transect.depth_avg(adcp,'U')    
    track['u_avg']=U_depth_avg.values[:,0]
    track['v_avg']=U_depth_avg.values[:,1]

    # Nearest surface valid velocity:
    def first_valid(x):
        x=x[np.isfinite(x)]
        if len(x):
            return x[0]
        else:
            return np.nan

    track['u_surf']= [ first_valid( adcp.U.values[samp,:,0] ) for samp in adcp.sample]
    track['v_surf']= [ first_valid( adcp.U.values[samp,:,1] ) for samp in adcp.sample]
    track['tnum'] = utils.to_unix( adcp.time.values + adcp_time_offset_to_utc )
    adcp.close()
    return track

##

# # Quick plot to see if that makes sense.
# # seems fine.
# plt.figure(3).clf()
# plt.quiver( track.x, track.y, track.u_avg, track.v_avg,color='b')
# plt.quiver( track.x, track.y, track.u_surf, track.v_surf,color='orange')

##

recs=[]

for fn in adcp_fns:
    rec=dict(adcp_src=fn)
    rec['track']=adcp_to_track(fn)
    recs.append(rec)
    
adcps=pd.DataFrame(recs)

##

track_common.dump_to_folder(adcps,'adcp_2018')
