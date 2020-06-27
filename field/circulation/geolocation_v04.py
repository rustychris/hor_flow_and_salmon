"""
Once pings have been matched, read pings and station locations in and
try to assemble trajectories.

This time, try Stan.

v04: look at 2018 data, first with an eye to detecting flows.
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
ds=xr.open_dataset('pings-2018-03-20T20:00_2018-03-20T22:00.nc')
# SM4, SM3
ds=ds.isel(rx=[2,3])


##
beacons=ds.rx_beacon.values
beacon_to_xy=dict( [ (ds.rx_beacon.values[i],
                      (ds.rx_x.values[i],ds.rx_y.values[i]))
                     for i in range(ds.dims['rx'])] )

## 

# can get a little bit of constraint from 2-rx pings, but
# not that useful yet.
is_multirx=(np.isfinite(ds.matrix).sum(axis=1)>1).values

# this is more useful for this stage
is_triplerx=(np.isfinite(ds.matrix).sum(axis=1)>2).values

# for testing, additionally limit to beacon pings, and only when the
# rx saw its own tx.
is_beacon=np.array( [ tag in beacon_to_xy for tag in ds.tag.values ] )
is_selfaware=np.zeros_like(is_beacon)
for i,tag in enumerate(ds.tag.values):
    if tag not in beacon_to_xy:
        continue
    else:
        # And did the rx see itself?
        local_rx=np.nonzero( ds.rx_beacon.values==tag )[0]
        if local_rx.size and np.isfinite( ds.matrix.values[i,local_rx[0]]):
            is_selfaware[i]=True

##

def model_inputs(sel_pings,xyt0=None):
    ds_strong=ds.isel(index=sel_pings)

    matrix=ds_strong.matrix.values
    rx_xy=np.c_[ds_strong.rx_x.values,
                ds_strong.rx_y.values]


    # Clock shifts for all but the first rx (assumes there are good hits for the 1st rx)
    shift_src=slice(0,ds_strong.dims['rx']-1)

    # for all pings, both tag and beacon
    # x coordinate, y coordinate, and tnum
    # the known values are filled in, and the nan's define the values to be
    # searched
    ping_xyt_template=np.nan*np.zeros( (ds_strong.dims['index'],3), np.float64)
    
    self_echo_count=0

    for i in range(ds_strong.dims['index']):
        tag=ds_strong.tag.values[i]
        if tag in beacon_to_xy:
            ping_xyt_template[i,:2]=beacon_to_xy[tag]
            # And did the rx see itself?
            local_rx=np.nonzero( ds_strong.rx_beacon.values==tag )[0]
            if local_rx.size:
                # HERE - this actually has to get the shift of the rx.
                # does fmin_powell do any better when this is left nan?
                # ping_xyt_template[i,2]=ds_strong.matrix.values[i,local_rx[0]]
                if np.isfinite(ping_xyt_template[i,2]):
                    self_echo_count+=1
            else:
                ping_xyt_template[i,2]=np.nan
        else:
            ping_xyt_template[i,:]=np.nan

    n_ping_xyt_unknown=np.isnan(ping_xyt_template).sum()

    if xyt0 is None:
        # origin offset for position and time
        xyt0=np.zeros(3,np.float64)
        xyt0[:2]=np.nanmean(ping_xyt_template[:,:2],axis=0)
        xyt0[2]=np.nanmean(np.nanmean(ds_strong.matrix.values))

    # 622 tag coordinates.
    #print("ping_xyt: %d known coordinates/times, %d unknown coordinates/time"%(np.isfinite(ping_xyt_template).sum(),
    #                                                                           np.isnan(ping_xyt_template).sum()))

    # At this point we have 11 dofs for clock shift.
    # and 986 dofs for tag position and time.
    return xyt0,ping_xyt_template,ds_strong,rx_xy


# would like to do this:
# sel_pings=is_triplerx | (is_multirx&is_beacon)
# But to keep it a bit more tractable right now, do this:
# sel_pings=is_triplerx
# this is for just getting clock offsets
# sel_pings=is_beacon&is_multirx

##

code = """
data {
 int<lower=1> Np; // number of pings
 int<lower=1> Nb; // number of beacons
 int<lower=1> Ndist; // number of unique distances
 
 real rx_t[Np,Nb];
 real<lower=0.0> sigma_t;
 real<lower=0.0> rx_c[Np,Nb];

 int<lower=1,upper=Nb> tx_beacon[Np];
 // allow 0 to catch indexing errors
 int<lower=0,upper=Ndist> dist_k[Nb,Nb];
}
transformed data {
 // receive time *without* t_shifts for either side,
 // relative to transmit time
 real rx_t_rel[Np,Nb];

 for ( p in 1:Np ) {
   for ( b in 1:Nb ) {
     rx_t_rel[p,b]=rx_t[p,b] - rx_t[p,tx_beacon[p]];
   }
 }
}
parameters {
 // only sample the unconstrained time shifts
 real t_shift_unknown[Nb-1];

 // just tune the upper right part of the triangle
 // diagonal is zero.
 real<lower=1.0,upper=10000> ur_dist[Ndist];
}
transformed parameters {
 real t_shift[Nb];

 t_shift[1]=0.0;
 for(b in 2:Nb) {
   t_shift[b]=t_shift_unknown[b-1];
 } 
}
model {
  // Are the priors screwing things up?

  // for(b in 2:Nb) {
  //   // 20 seconds
  //   t_shift_unknown[b-1] ~ normal(0,20000); // weak priors for shifts
  // }
  // 
  // for(b in 1:Ndist) {
  //   // weak prior for distances
  //   ur_dist[b] ~ normal(100,1000);
  // }

  for ( p in 1:Np ) {
    for ( b in 1:Nb ) {
      // but only select this when rx_t[p,b] is not nan
      int a=tx_beacon[p];
      if ( (!is_nan(rx_t[p,b])) && (a!=b) ) { 
        // try to make it closer to obs ~ distribution(parameters)
        //  I thought the sign on the shifts should be the other way around...?
        // rx_t_rel[p,b] ~ normal(ur_dist[dist_k[a,b]]/rx_c[p,b]
        //                        - t_shift[a]
        //                        + t_shift[b],
        //                        sigma_t);

        target += normal_lpdf( rx_t_rel[p,b] | 
                               ur_dist[dist_k[a,b]]/rx_c[p,b]
                               - t_shift[a]
                               + t_shift[b],
                               sigma_t);


        print("--normal--");
        print("rx_t_rel=",rx_t_rel[p,b]);
        print("transit time ms=",ur_dist[dist_k[a,b]]/rx_c[p,b]);
        print("time shifts: ",-t_shift[a],"  ",t_shift[b]);
        print("----");
      }
    }
  }
}
"""

# This is surprisingly slow
sm = pystan.StanModel(model_code=code)

##

with open('model.cpp','wt') as fp:
    fp.write(sm.model_cppcode)
    

##

# What about variation in speed of sound?
# Does it need to be in 3D?

# this should get the same answer, but I'm gettin tx_x that appear
# to be scaled up by this much
time_scale=1000 # model time is 1000x actual time

def build_data(ping_xyt_template,ds_strong,xyt0,rx_xy):
    matrix=ds_strong.matrix.values
    temps=ds_strong.temp.values

    tx_beacon=np.zeros(matrix.shape[0],np.int32)
    for p,tag in enumerate(ds_strong.tag.values):
        tx_beacon[p] = 1+np.nonzero(ds_strong.rx_beacon.values==tag)[0][0]

    data=dict(Np=matrix.shape[0],
              Nb=matrix.shape[1],
              rx_t=(matrix-xyt0[2])*time_scale,
              rx_c=seawater.svel(0,ds_strong['temp'],0) / time_scale,
              rx_x=rx_xy[:,0]-xyt0[0],
              rx_y=rx_xy[:,1]-xyt0[1],
              tx_x=ping_xyt_template[:,0]-xyt0[0],
              tx_y=ping_xyt_template[:,1]-xyt0[1],
              # tx_t=ping_xyt_template[:,2]-xyt0[2],
              tx_beacon=tx_beacon,
              sigma_t=0.0005*time_scale, # 0.5ms timing precision
              )

    # explicitly pass in the indexing for matrix to linear
    Nb=data['Nb']
    dist_k=np.zeros( (Nb,Nb), np.int32 ) 
    for ai in range(Nb):
        for bi in range(ai+1,Nb):
            a=ai+1
            b=bi+1
            k = (Nb*(Nb-1)/2) - (Nb-a+1)*(Nb-a)/2 + b-a;
            dist_k[ai,bi]=dist_k[bi,ai]=k
    data['dist_k']=dist_k
    data['Ndist']=(Nb*(Nb-1))//2
    return data


mask=is_beacon&is_multirx
# mask=np.nonzero(mask)[0][:2] # just the first two pings, which includes one of each direction

xyt0,ping_xyt_template,ds_strong,rx_xy=model_inputs(mask)
data=build_data(ping_xyt_template,ds_strong,xyt0,rx_xy)

data['sigma_t']=1
op_shifts=sm.optimizing(data=data,iter=1000,tol_rel_grad=1e-2, # tol_rel_obj=0.01,
                        algorithm='Newton',
                        init=lambda: dict(ur_dist=np.array([2.0]),
                                          t_shift=[0.,-2787.]))

# t_shift: comes out about [0,2458 ms]
# ur_dist: pegged to 1000
# tx_t: 
# clock_time is coming out 0 -256
#  or 400.899 0

# With a single ping, it lets distance go to 0, tx_t is 1357
# and t_shift is 0.271s
# seems that it has preferred making distance very short rather
# than adjusting t_shift

# it's currently at least getting the transit times symmetric.
# it tries to drive ur_dist to 0, and ends up with a LL of -5200.
# what if I constrain

# What I *expect*, from the manual approach below, is a
# transit time of 72.1ms
# t_shift of [0,-2787ms]
# distance of 48.5m

# I'm getting a t_shift of the opposite sign...
# and distance pegged at lower bound.

# printing the values going into the distribution, it still
# seems like the probability would be improved by
# making the distance longer.
# --normal-- # This one would fit better if transit time were longer
# rx_t_rel=-2715.18
# transit time ms=0.672287
# time shifts: -0  -2787.32
# ----
# --normal-- # This one would also fit better if transit time were longer
# rx_t_rel=2859.47
# transit time ms=0.672553
# time shifts: 2787.32  0
# ----

# Using Newton, I at least get some reasonable values.
# ur_dist comes out as 107.3m.
# For some reason lBFGS is not as stable.

## How close is that really?

for p in range(data['Np']):
    a=data['tx_beacon'][p]-1
    for b in range(data['Nb']):
        if a==b: continue

        d=np.atleast_1d(op_shifts['ur_dist'])[data['dist_k'][a,b]-1]
        transit_dt=d/data['rx_c'][p,b]

        clock_dt=( (data['rx_t'][p,b]-op_shifts['t_shift'][b])
                   -
                   (data['rx_t'][p,a]-op_shifts['t_shift'][a]) )
        print(f"ping {p}: transit_dt: {transit_dt:.3f}ms  clock_dt: {clock_dt:.3f} ms  error: {transit_dt-clock_dt:.3f} ms")


# Not bad.
# need a higher order t_shift to get the errors down, though.
# HERE: consider next step:
#  1: Add a linear drift term, see if the errors are approaching microseconds.
#  2: Abandon STAN and continue on the direct approach below
#  3: Add an additional station, start to look for circulation.

##

# so how is it getting such bad results?
# do the calculation manually for two rxs:

# Matrix of dists
t_shifts=np.zeros(ds_strong.dims['rx'])
t_shifts[0]=0.0

# make a guess of t_shifts[1] based on
# average of a->b and b->a
src_mask= data['tx_x'][:,None] == data['rx_x'][None,:]
assert np.all(src_mask.sum(axis=1)==1) # somebody has to hear it

# how much 'later' b saw it than a
deltas=data['rx_t'][:,1] - data['rx_t'][:,0]

dt_ab=np.mean(deltas[src_mask[:,0]]) 
dt_ba=np.mean(deltas[src_mask[:,1]])

# symmetric mean to avoid a bias
dt_sym=0.5*(dt_ab+dt_ba) # mix of clock offset and travel asymmetry
dt_transit=0.5*(dt_ab-dt_ba) # mean transit time


matrix=ds_strong.matrix
Np,Nb=matrix.shape
transits=np.zeros( (Nb,Nb), dtype=[ ('transit_mean',np.float64),
                                    ('transit_std',np.float64),
                                    ('n',np.int32)])
transits['transit_mean']=np.nan

t_shifts[1]=dt_sym

# this approach forces symmetry and 0 time to self
transits['transit_mean'][0,1]=dt_transit
transits['transit_mean'][1,0]=dt_transit
transits['transit_mean'][0,0]=0.0
transits['transit_mean'][1,1]=0.0

# How does that work out for speed of sound?
# Not good.  Distance is about 30m.
dist_a_b=np.sqrt( (ds_strong.rx_x.values[1] - ds_strong.rx_x.values[0])**2 +
                  (ds_strong.rx_y.values[1] - ds_strong.rx_y.values[0])**2)
c_mean=data['rx_c'].mean()

exp_transit_mean = dist_a_b/c_mean

# Something is quite off.  Maybe the time shifts aren't linear?
# transit mean should be 20ms, but I'm getting 72ms.
##

fig=plt.figure(1)
fig.clf()
# bimodal.
sel=src_mask[:,0]
# these trend smoothly over 0.11 ms, mean of -2715
plt.plot(ds_strong.matrix.values[sel,0],deltas[sel],label='a-b deltas')
sel=src_mask[:,1]
# these smoothly trend over 0.18 ms, mean of -2859
plt.plot(ds_strong.matrix.values[sel,1],deltas[sel],label='b-a deltas')
plt.legend(loc='upper left')
fig.tight_layout()

# So either the coordinates are wrong for the receivers, or somewhere
# along the way the receivers got out of order.
# probably the latter.
# ds gives these as SM4 and SM3

# Back track some specific ping times.
# SM4, first ping of this period:
#  4132811.186296
# there shouldn't be any time_scale in there.
# that translates back to 2018-03-20T20:00:11.186296000
# should be tag FF18
# that time does not show up in the DET file for SM4.
# but one 8h earlier does show up.
# okay -- I'm using UTC, and the fields are for PST.
# fine.
# so this does show up in SM4
# and what about SM3?
# x=ds_strong.matrix[0,1] ; pm.T0 + x*1e9*np.timedelta64(1,'ns')
#  => 2018-03-20T20:00:08.471121000
# and that does show up in SM3.

# the coordinates match the already plotted set of coordinates.
# so I don't think it's an error in how I've plotted or processed
# the spatial information.

# at this point, can either try to reformulate the fits to estimate
# the distance, or wait and ask Mike on Thursday.

##
def calc_transits(ds_strong,op_shifts,time_scale=time_scale):
    # For each pair of beacons with nonzero pings, plot
    # A=>B and B=>A stats
    matrix=ds_strong.matrix
    Np,Nb=matrix.shape
    transits=np.zeros( (Nb,Nb), dtype=[ ('transit_mean',np.float64),
                                        ('transit_std',np.float64),
                                        ('n',np.int32)])
    transits['transit_mean']=np.nan

    for tx_A,beacon_tag in enumerate(ds_strong.rx_beacon.values):
        shift_A=op_shifts['t_shift'][tx_A]
        for rx_B,tag_B in enumerate(ds_strong.rx):
            shift_B=op_shifts['t_shift'][rx_B]

            # pings from A 
            tx_sel=(ds_strong.tag==beacon_tag)
            # that A heard
            tx_sel=tx_sel & np.isfinite( matrix[:,tx_A] )
            # and were received at B
            rx_sel=np.isfinite( matrix[:,rx_B] )
            sel=tx_sel&rx_sel
            if sel.sum()==0: continue

            # Get a mean transit time
            # tx_times=matrix[sel,tx_A]+shift_A/time_scale
            tx_times=op_shifts['tx_t'][sel]/time_scale + xyt0[2]
            rx_times=matrix[sel,rx_B]+shift_B/time_scale
            transit_times=rx_times-tx_times
            transits['transit_mean'][tx_A,rx_B]=np.mean(transit_times)
            transits['transit_std'][tx_A,rx_B]=np.std(transit_times)
            transits['n'][tx_A,rx_B]=len(transit_times)
    return transits

transits=calc_transits(ds_strong,op_shifts)

##

# Summarize the fit
zoom=(647097., 647527., 4185668., 4185973)
plt.figure(2).clf()
fig,ax=plt.subplots(figsize=(12,10),num=2)

img.plot(zorder=-10,ax=ax)
ax.axis('off')
fig.tight_layout()

# RX locations
ax.plot( ds.rx_x,ds.rx_y,'go',zorder=5,ms=12)
ax.axis(zoom)

# print summary of the per-station shifts
station_df=pd.DataFrame()
station_df['beacon']=ds.rx_beacon
station_df['t_shift_s']=op_shifts['t_shift'] / time_scale
print(station_df)

# annotate the stations
for a in range(ds_strong.dims['rx']):
    ax.text( ds_strong.rx_x.values[a],
             ds_strong.rx_y.values[a],
             "%d: %s"%(a,ds.rx.values[a]),
             va='center',ha='center',
             zorder=10,color='w')

for a in range(ds_strong.dims['rx']):
    for b in range(ds_strong.dims['rx']):
        if a==b: continue
        
        t=transits['transit_mean'][a,b]
        n=transits['n'][a,b]
        # if np.isnan(t): continue
        if n<20: continue

        idxs=[a,b]
        l=ax.plot( ds_strong.rx_x.values[idxs],
                   ds_strong.rx_y.values[idxs],
                   'k-',lw=np.sqrt(n)/3)[0]
        plot_utils.annotate_line(l,"%.3f"%(1000*t))
        
##

