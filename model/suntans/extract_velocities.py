"""
Check on long run and develop code to extract point velocity estimates.
"""
import matplotlib.pyplot as plt
from stompy.model.suntans import sun_driver
from stompy.plot import plot_utils
from stompy.grid import unstructured_grid
import pandas as pd

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
detects=pd.read_csv("../../field/tags/cleaned_half meter.csv",
                    parse_dates=['dt'])
# for this we don't need better than minute precision.

## 
detect_vel=np.nan*np.zeros( (len(detects),2), np.float64)

##

seq=model.chain_restarts()

##
# HERE - Need to confirm time zones, and if dt and epoch_secs
# are consistent.

six.moves.reload_module(sun_driver)
mod=sun_driver.SuntansModel.load('/home/rusty/src/hor_flow_and_salmon/model/suntans/runs/snubby_cfg001_20180316')

all_stations_ds=[]

# for model_day in [mod]:
for model_day in seq:
    print(model_day) # HERE gets really bogged down on day 2.
    detect_this_day=(detects.dt>=model_day.run_start) & (detects.dt<=model_day.run_stop)
    idxs=np.nonzero(detect_this_day.values)[0]

    if len(idxs)==0: continue

    xy=np.c_[detects.X_UTM.values[idxs],
             detects.Y_UTM.values[idxs]]

    # These will have the whole day
    stations=model_day.extract_station(xy=xy)
    # Slim to just the specific timestamps
    detect_times=detects.dt.values[idxs]
    time_idxs=np.searchsorted( stations.time, detect_times )

    station_dss=[ stations.isel(stndim0=station_i,time=time_idx)
                  for station_i,time_idx in enumerate(time_idxs)]
    stations_ds=xr.concat(station_dss,dim='station')
    # And record the original index so they can be put back together
    stations_ds['detect_idx']=('station',),idxs
    all_stations_ds.append(stations_ds)

##
if 0: # pickable map with station timeseries
    fig=plt.figure(2)
    fig.clf()
    ax=fig.add_axes([0,0,1,1])
    model.grid.plot_edges(lw=0.5,ax=ax)
    plt.setp(ax.get_xticklabels(),visible=0)
    plt.setp(ax.get_yticklabels(),visible=0)

    scat=ax.scatter(stations.station_x.values,
                    stations.station_y.values,
                    30,color='g')

    ax_t=fig.add_axes([0.1,0.14,0.35,0.35])

    def plot_station_ts(idx,**kw):
        ax_t.cla()
        ax_t.plot( stations.time, stations.uc.isel(stndim0=idx,Nk=0), label="U")
        ax_t.plot( stations.time, stations.vc.isel(stndim0=idx,Nk=0), label="V")
        ax_t.legend()
        ax_t.figure.canvas.draw()

    plot_utils.enable_picker(scat,cb=plot_station_ts)
