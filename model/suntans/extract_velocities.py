"""
Check on long run and develop code to extract point velocity estimates.
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

import six
six.moves.reload_module(unstructured_grid)
six.moves.reload_module(sun_driver)

##

model=sun_driver.SuntansModel.load("runs/snubby_cfg001_20180411")

##

if 1: # plot a quick a test of the output
    test_xy=[647342., 4185758.]

    ds=model.extract_station(xy=test_xy,chain_count=0)


    fig=plt.figure(1)
    fig.set_size_inches([8,6],forward=True)
    fig.clf()
    ax=fig.add_axes([0,0,1,1])

    model.grid.plot_edges(lw=0.5,ax=ax)
    plt.setp(ax.get_xticklabels(),visible=0)
    plt.setp(ax.get_yticklabels(),visible=0)

    ax_t=fig.add_axes([0.1,0.14,0.35,0.35])

    ax_t.plot( ds.time,ds.uc.isel(Nk=0),label="U" )
    ax_t.plot( ds.time,ds.vc.isel(Nk=0),label="V" )
    ax_t.legend()
    plt.setp(ax_t.get_xticklabels(),rotation=40,ha='right')
    ax_t.set_ylabel('Vel (m/s)')

    ax.plot([test_xy[0]],
            [test_xy[1]],
            'ro',ms=7,mec='k',mew=0.6)
    dx=90
    ax.axis([646965-dx, 647758.-dx, 4185450., 4186045.])
    # fig.savefig('centerline_velocity_2018.png')

##

# extract a velocity for each of the detections.
#inp_fn="../../field/tags/segments_2.csv"
inp_fn="../../field/tags/segments_2m.csv"
segments=pd.read_csv(inp_fn)

##

# first line of cleaned_half meter.csv:
# 1,71aa,3,647410.6,4185684.1,5,1521903581,6.20E-05,3/24/2018 7:59,878668

# epoch20180324T0759=1521903581
# datetime.datetime.fromtimestamp(epoch20180324T0759) # datetime.datetime(2018, 3, 24, 7, 59, 41)
# # daylight savings time began on March 11, 2018.
# # subtract 20 days from the above time stamp, convert:
# datetime.datetime.fromtimestamp(epoch20180324T0759-20*86400) # datetime.datetime(2018, 3, 4, 6, 59, 41)
# # does this conversion change around PST/PDT? yes.
# 
# (1521903581 % 86400) / 3600. # ==> 14.994  i.e. UTC 14:59:41
# # so that suggests 7 hour offset from UTC, which is PDT

# HERE - either get info from Ed, or consult the data to figure out the timezone for
# Ed's datenums.

# dt_utc=[datetime.datetime.utcfromtimestamp(es)
#         for es in detects.Epoch_Sec.values ]
# 
# detects['dnum_utc']=detects_dnum_utc=utils.to_dnum(dt_utc)

# segment dnums appear to be in PDT.
# with some quantization of about 30 seconds.
seg_dnum_to_utc=7./24 # add this to segment dnums to get utc.

##

segment_vel=np.nan*np.zeros( (len(segments),2), np.float64)

##

# And what timezone is the model in? UTC.


six.moves.reload_module(sun_driver)
mod=sun_driver.SuntansModel.load('/home/rusty/src/hor_flow_and_salmon/model/suntans/runs/snubby_cfg001_20180411')
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
    # break
    # Why is this so slow? b/c it loops over variables, then  stations,
    # and uses two isel() calls in the inner loop
    stations=model_day.extract_station(xy=sample_xy)
    # Slim to just the specific timestamps
    sample_times=seg_dt64_utc[idxs]

    # this gives time errors [0,1800s]
    #time_idxs=np.searchsorted( stations.time.values, sample_times )
    # should give time errors [-900s,900s]
    time_idxs=utils.nearest(stations.time.values, sample_times )

    station_dss=[ stations.isel(station=station_i,time=time_idx)
                  for station_i,time_idx in enumerate(time_idxs)]
    stations_ds=xr.concat(station_dss,dim='station')
    # And record the original index so they can be put back together
    stations_ds['detect_idx']=('station',),idxs
    all_stations_ds.append(stations_ds)

##

# Still kind of slow, but workable.
#  yields ds with station len 18792
#  
joined=xr.concat(all_stations_ds,dim='station')
# Make sure we hit every station
assert np.all(np.unique(all_idxs)==np.arange(len(all_idxs)))
# And reorder to get the same order as the inputs
joined=joined.sortby(joined.detect_idx)
assert joined.dims['station']==len(segments)

# For 2D, pick Nk=0
joined=joined.isel(Nk=0)

##
if 1: # quiver with all stations
    fig=plt.figure(2)
    fig.clf()
    ax=fig.add_axes([0,0,1,1])
    model.grid.plot_edges(lw=0.5,ax=ax)
    plt.setp(ax.get_xticklabels(),visible=0)
    plt.setp(ax.get_yticklabels(),visible=0)

    # scat=ax.scatter(joined.station_x.values,
    #                 joined.station_y.values,
    #                 30,color='g')
    quiv=ax.quiver(joined.station_x.values,
                   joined.station_y.values,
                   joined.uc.values,
                   joined.vc.values,
                   joined.distance_from_target.values,
                   cmap='jet')

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

##
if 0: # time error
    # shows errors distributed evenly [-900,900s]
    time_error = (utils.to_dnum(joined.time) - segments.dnm.values) - 7./24
    plt.figure(3).clf() ; plt.hist( time_error * 86400, 100)

##

out_fn=inp_fn.replace(".csv","-model20190203.csv")
assert out_fn!=inp_fn

segments.to_csv(out_fn,index=False)

