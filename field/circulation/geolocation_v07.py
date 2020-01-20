"""
Once pings have been matched, read pings and station locations in and
try to assemble trajectories.

v04: look at 2018 data, first with an eye to detecting flows.
v05: build on v04, adding linear drift term
v06: for a promising group of 3 stations, try to fit circulation.
v07: revert to direct estimation rather than STAN, as there is too much noise
     to fit just the linear drift

"""

import pandas as pd
import xarray as xr
from stompy import utils
import numpy as np
import pystan
import matplotlib.pyplot as plt
from stompy.spatial import field
import seawater

from stompy.plot import plot_utils
##

img=field.GdalGrid("../../model/grid/boundaries/junction-bathy-rgb.tif")

## 

try:
    ds_tot.close()
except NameError:
    pass
ds_tot=xr.open_dataset('pings-2018-03-20T20:00_2018-03-20T22:00.nc')


# Swapping is now handled by an edit to the csv.  see prepare_pings.
# # Try swapping the coordinates of sm2 and sm3.
# # That agrees much better with the calculated distances.
# # This is now handled 
# l1='SM2'
# l2='SM3'
# li1=np.nonzero(ds_tot.rx.values==l1)[0][0]
# li2=np.nonzero(ds_tot.rx.values==l2)[0][0]
# 
# for field in ['rx_x','rx_y','rx_z']:
#     v1=ds_tot[field].values[li1]
#     v2=ds_tot[field].values[li2]
#     ds_tot[field].values[li1]=v2
#     ds_tot[field].values[li2]=v1


## 

# Simplify this down to taking two stations and calculating a time
# series of the effective clock offset and associated uncertainty.
def effective_clock_offset(rxs,ds_tot):
    ds=ds_tot.sel(rx=rxs)
    # weird. sometimes adds an extra array(..,dtype=object) layer
    ds['rx_beacon']=('rx',),[ds_tot.rx_beacon.sel(rx=rx).item() for rx in rxs]

    beacon_to_xy=dict( [ (ds.rx_beacon.values[i],
                          (ds.rx_x.values[i],ds.rx_y.values[i]))
                         for i in range(ds.dims['rx'])] )

    # both beacons heard it
    is_multirx=(np.isfinite(ds.matrix).sum(axis=1)>1).values
    # it came from one of them.
    is_selfaware=np.zeros_like(is_multirx)
    for i,tag in enumerate(ds.tag.values):
        if tag not in beacon_to_xy:
            continue
        else:
            # And did the rx see itself?
            local_rx=np.nonzero( ds.rx_beacon.values==tag )[0]
            if local_rx.size and np.isfinite( ds.matrix.values[i,local_rx[0]]):
                is_selfaware[i]=True

    sel_pings=is_multirx&is_selfaware

    ds_strong=ds.isel(index=sel_pings)

    matrix=ds_strong.matrix.values
    rx_xy=np.c_[ds_strong.rx_x.values,
                ds_strong.rx_y.values]

    temps=ds_strong.temp.values
    tx_beacon=np.zeros(matrix.shape[0],np.int32)
    for p,tag in enumerate(ds_strong.tag.values):
        tx_beacon[p] = np.nonzero(ds_strong.rx_beacon.values==tag)[0][0]

    rx_c=seawater.svel(0,ds_strong['temp'],0) / time_scale
    # occasional missing data...
    rx_c=utils.fill_invalid(rx_c,axis=1)

    # do the calculation manually for two rxs:

    ab_mask=(tx_beacon==0)
    ba_mask=(tx_beacon==1)

    assert np.all(ab_mask|ba_mask) # somebody has to hear it

    # how much 'later' b saw it than a
    t=matrix[:,0] # take rx 0 as the reference

    # without rx_c, transit time varies from 0.051970 to 0.051945
    # for a variation of 0.48ppt
    # with rx_c, transit time distance varies 0.07721 to 0.07710
    # for a variation of 1.4ppt, and it introduces some step changes.
    # so for this application it's better to have the smaller variation
    # that also changes smoothly, rather than correct back to a precise
    # distance
    deltas=(matrix[:,1] - matrix[:,0]) # * rx_c.mean(axis=1)

    # partial time series
    dt_ab=deltas[ab_mask]
    dt_ba=deltas[ba_mask]

    dt_ab_dense=np.interp(t,
                          t[ab_mask],deltas[ab_mask])
    dt_ba_dense=np.interp(t,
                          t[ba_mask],deltas[ba_mask])
    # proxy for the uncertainty.  Not scaled!
    dt_ab_dense_std=np.abs(t-utils.nearest_val(t[ab_mask],t))
    dt_ba_dense_std=np.abs(t-utils.nearest_val(t[ba_mask],t))

    dt_offset =0.5*(dt_ab_dense+dt_ba_dense) # sum of clock offset and travel asymmetry
    dt_transit=0.5*(dt_ab_dense-dt_ba_dense) # transit time
    dt_std=dt_ab_dense_std+dt_ba_dense_std

    ds=xr.Dataset()
    ds['time']=('time',),t
    ds['offset']=('time',),dt_offset
    ds['transit']=('time',),dt_transit
    ds['error']=('time',),dt_std
    ds['c']=('time',),rx_c.mean(axis=1)
    ds['rx']=('rx',),rxs

    return ds

##

def plot_circulation(rxs,ds_tot,num=1):
    ds_ab=effective_clock_offset([rxs[0],rxs[1]],ds_tot)
    ds_bc=effective_clock_offset([rxs[1],rxs[2]],ds_tot)
    ds_ca=effective_clock_offset([rxs[2],rxs[0]],ds_tot)

    # Combine
    t_common=np.unique( np.concatenate( (ds_ab.time.values,
                                         ds_bc.time.values,
                                         ds_ca.time.values) ))

    ds_abc=xr.Dataset()
    ds_abc['time']=('time',),t_common

    ds_abc['offset']=('time',), ( np.interp(t_common,ds_ab.time.values, ds_ab.offset.values)
                                  + np.interp(t_common,ds_bc.time.values, ds_bc.offset.values)
                                  + np.interp(t_common,ds_ca.time.values, ds_ca.offset.values))

    error=0
    for ds in [ds_ab,ds_bc,ds_ca]:
        near_idx=utils.nearest(ds.time,t_common)
        # The accumulated "error" is the error of the nearest value in each source
        error=error+ds['error'].values[near_idx]
        # *and* the time offset from that value.
        error += np.abs(t_common-utils.nearest_val(ds.time.values,t_common))
    ds_abc['error']=('time',error)

    plt.figure(num).clf()
    fig,axs=plt.subplots(4,1,num=num,sharex=True)
    for ds in [ds_ab,ds_bc,ds_ca]:
        rx_from,rx_to=ds.rx.values
        label="%s-%s"%(rx_from,rx_to)
        axs[0].plot(ds.time,ds.offset,label=label)
        axs[1].plot(ds.time,ds.transit,label=label)

    axs[0].set_ylabel('Offset')
    axs[1].set_ylabel('Transit')
    axs[2].plot(ds_abc.time,1e6*ds_abc.offset,label="Circ")
    axs[2].set_ylabel('Circulation (us)')
    axs[3].plot(ds_abc.time,ds_abc.error,label="Error")
    [ax.legend(loc='upper right') for ax in axs]

    return fig,ds_abc,[ds_ab,ds_bc,ds_ca]

#plot_circulation(['SM2','SM3','AM4'],ds_tot,num=1)
#plot_circulation(['SM2','SM3','AM9'],ds_tot,num=2) # bad
#plot_circulation(['SM2','SM3','SM4'],ds_tot,num=3)
#plot_circulation(['SM2','SM3','AM5'],ds_tot,num=4)
#plot_circulation(['SM3','SM2','AM5'],ds_tot,num=5)
#plot_circulation(['SM3','SM2','AM2'],ds_tot,num=6)
#plot_circulation(['AM1','AM2','AM9'],ds_tot,num=7) # Errors are pretty high.

# Overall, not too bad.
