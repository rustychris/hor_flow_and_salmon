"""
Extract point velocity estimates.
"""
import matplotlib.pyplot as plt
from stompy.model.suntans import sun_driver
from stompy.plot import plot_utils
from stompy import utils
from stompy.grid import unstructured_grid
import pandas as pd
import xarray as xr
import numpy as np

##

# extract a velocity for each of the detections.
# segment dnums appear to be in PDT.
# with some quantization of about 30 seconds.
# And what timezone is the model in? UTC.

inp_fn="../../field/tags/segments_2m.csv"
seg_dnum_to_utc=7./24 # add this to segment dnums to get utc.

segments=pd.read_csv(inp_fn)

# choose the last run of a sequence:
mod=sun_driver.SuntansModel.load('runs/snubby_cfg001_20180411')

seq=mod.chain_restarts()

all_stations_ds=[]

seg_dt64_utc=utils.to_dt64(segments.dnm.values+seg_dnum_to_utc)

for model_day in seq:
    hits_this_day=(seg_dt64_utc>=model_day.run_start) & (seg_dt64_utc<=model_day.run_stop)
    idxs=np.nonzero(hits_this_day)[0]

    if len(idxs)==0: continue
    print("%d samples to pull for %s -- %s"%(len(idxs),
                                             model_day.run_start,
                                             model_day.run_stop))
    
    sample_xy=np.c_[segments.xm.values[idxs],
                    segments.ym.values[idxs]]

    # These will have the whole day
    stations=model_day.extract_station_map(xy=sample_xy)
    # Slim to just the specific timestamps
    sample_times=seg_dt64_utc[idxs]

    # this gives time errors [0,1800s]
    #time_idxs=np.searchsorted( stations.time.values, sample_times )
    # this centers the time offsets [-900s,900s]
    time_idxs=utils.nearest(stations.time.values, sample_times )

    station_dss=[ stations.isel(station=station_i,time=time_idx)
                  for station_i,time_idx in enumerate(time_idxs)]
    stations_ds=xr.concat(station_dss,dim='station')
    # And record the original index so they can be put back together
    stations_ds['input_idx']=('station',),idxs
    all_stations_ds.append(stations_ds)

joined=xr.concat(all_stations_ds,dim='station')
# Make sure we hit every station
all_idxs=joined.input_idx.values.copy()
assert np.all(np.unique(all_idxs)==np.arange(len(all_idxs)))
# And reorder to get the same order as the inputs
joined=joined.sortby(joined.input_idx)
assert joined.dims['station']==len(segments)

# For 2D, pick Nk=0
joined=joined.isel(Nk=0)

##
if 1: # quiver with all stations
    fig=plt.figure(2)
    fig.clf()
    ax=fig.add_axes([0,0,1,1])
    model.grid.plot_edges(lw=0.5,ax=ax,color='k',zorder=-3)
    plt.setp(ax.get_xticklabels(),visible=0)
    plt.setp(ax.get_yticklabels(),visible=0)

    # scat=ax.scatter(joined.station_x.values,
    #                 joined.station_y.values,
    #                 30,color='g')
    umag=utils.mag(np.c_[joined.uc.values,joined.vc.values])
    quiv=ax.quiver(joined.station_x.values,
                   joined.station_y.values,
                   joined.uc.values,
                   joined.vc.values,
                   # umag,cmap='jet',
                   (joined.time.values-joined.time.values[0])/np.timedelta64(1,'D'),
                   cmap='jet',
                   scale=40.0)
    plt.colorbar(quiv,label="Time (d)")
    ax.quiverkey(quiv,0.1,0.1,0.5,"0.5 m/s")
    ax.axis( (647091.6894404562, 647486.7958566407, 4185689.605777047, 4185968.1328291544) )
    fig.savefig('model-extracted-velocities.png')
    
##
# Add back to original dataframe and output
segments['model_time']=joined['time'].values
segments['model_z_bed']=-joined['dv'].values
segments['model_z_eta']=joined['eta'].values
segments['model_u']=joined['uc'].values
segments['model_v']=joined['vc'].values
segments['model_loc_err']=joined['distance_from_target'].values
segments['model_x']=joined['station_x'].values
segments['model_y']=joined['station_y'].values


if 0: # time error
    # shows errors distributed evenly [-900,900s]
    time_error = (utils.to_dnum(joined.time) - segments.dnm.values) - 7./24
    plt.figure(3).clf() ; plt.hist( time_error * 86400, 100)


out_fn=inp_fn.replace(".csv","-model20190314.csv")
assert out_fn!=inp_fn

segments.to_csv(out_fn,index=False)

