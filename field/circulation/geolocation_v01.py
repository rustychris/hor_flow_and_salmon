"""
Once pings have been matched, read pings and station locations in and
try to assemble trajectories.
"""

import pandas as pd
import xarray as xr
from stompy import utils
import numpy as np
import matplotlib.pyplot as plt

##

ds=xr.open_dataset('pings-2019-04-14T20:00_2019-04-14T22:00.nc')

##
beacons=ds.rx_beacon.values
beacon_to_xy=dict( [ (ds.rx_beacon.values[i],
                      (ds.rx_x.values[i],ds.rx_y.values[i]))
                     for i in range(ds.dims['rx'])] )



## 
# First, just see how things go with a direct optimization approach.

# The solution space:
#  For starters, assume no clock drift during the time span of
#  the data.

# Should slim the dataset some.  1416 of the pings were received by
# exactly 1 rx.  Those are of no use at this stage. This gets us down
# to 622 tag coordinates.

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

# sel_pings=is_multirx&is_beacon&is_selfaware
sel_pings=is_triplerx
ds_strong=ds.isel(index=sel_pings)

# vec

vec_len=0

# Clock shifts for all but the first rx (assumes there are good hits for the 1st rx)
vec_len+=ds_strong.dims['rx']-1
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
ping_xyt_src=slice(vec_len,vec_len+n_ping_xyt_unknown)
ping_xyt_dst=np.isnan(ping_xyt_template)
vec_len+=n_ping_xyt_unknown

# origin offset for position and time
xyt0=np.zeros(3,np.float64)
xyt0[:2]=np.nanmean(ping_xyt_template[:,:2],axis=0)
xyt0[2]=np.nanmean(np.nanmean(ds_strong.matrix.values))
ping_xyt_template_shifted=ping_xyt_template-xyt0[None,:]

# 622 tag coordinates.
print("ping_xyt: %d known coordinates/times, %d unknown coordinates/time"%(np.isfinite(ping_xyt_template).sum(),
                                                                           np.isnan(ping_xyt_template).sum()))

# At this point we have 11 dofs for clock shift.
# and 986 dofs for tag position and time.

##

# vec: the search vector processed by the optimization routine
# params: structured arrays ready for calculating cost, plotting, etc.
#    include any filling, scaling, offsets, etc. to go from values
#    that are "nice" for the optimization to values that are "nice"
#    for the physical problem.

# packing and unpacking
def vec_to_params(vec):
    """
    vec: ndarray of vec_len float elements
    returns 
    shifts,ping_xyt
    """
    shifts=vec[shift_src]
    ping_xyt=ping_xyt_template_shifted.copy()
    ping_xyt[ping_xyt_dst]=vec[ping_xyt_src]
    ping_xyt+=xyt0[None,:]
    return shifts,ping_xyt

def params_to_vec(shifts,ping_xyt):
    """
    vec: ndarray of vec_len float elements
    returns 
    shifts,ping_xyt
    """
    vec=np.zeros(vec_len,np.float64)
    vec[shift_src]=shifts
    vec[ping_xyt_src]=(ping_xyt-xyt0[None,:])[ping_xyt_dst]
    return vec
    
    
# Initial guess
shifts_init,ping_xyt_init=vec_to_params(np.zeros(vec_len,np.float64))

shifts_init[:]=5.0 # typical scale for clock offset
# ping locations all default to the center, plus 50 to give a scale
# after shifting.  Note that positions set here for beacon pings
# are ignored by params_to_vec, so this is effectively only for
# the tag pings.
ping_xyt_init[:,:2]=xyt0[:2] + np.array([50,50])

# times come from min of observed times.  For self-received
# beacon pings, can't just use the time at the beacon because
# ping times are in global absolute time (i.e. rx 0 time),
# but the beacon time is local time.
ping_xyt_init[:,2]=np.nanmin(ds_strong.matrix.values,axis=1)
vec_init=params_to_vec(shifts_init,ping_xyt_init)

##

matrix=ds_strong.matrix.values
rx_xy=np.c_[ds_strong.rx_x.values,
            ds_strong.rx_y.values]

## 
# Cost function
vec=vec_init

# shifts are added to times in matrix to get "real" times
# currently this is about 2ms.
# after fixing the bug with the ping times, it's getting
# much better.  costs now in the 1e5 range.

def cost(vec):
    shifts,ping_xyt=vec_to_params(vec)

    adj_times=matrix.copy()
    adj_times[:,1:] += shifts[None,:]

    transit_times=adj_times - ping_xyt[:,2,None]
    c=1500 # constant speed of sound for the moment.
    transit_dist=transit_times*c

    #         (2344, 2)       (12,2)
    geo_dists=utils.dist( ping_xyt[:,None,:2] - rx_xy[None,:] )

    errors=(transit_dist - geo_dists)

    mse=np.nanmean(errors**2) # m^2 errors
    cost=mse
    
    # So being off by 100m is 1e4 error, same as having
    # a -10ms travel time.
    
    # bad_transit=transit_times[np.isfinite(transit_times)]
    # neg_transit_cost=(bad_transit.clip(None,0)**2 * 1e8).sum()
    # cost+=neg_transit_cost
    
    #print("%.2f"%mse)
    return cost

# OPTIMIZE
from scipy.optimize import fmin_powell, fmin, basinhopping,brute

##

# when just solving for beacon pings:
#   Converges at cost=7.37, with no adjustments for c.
# when solving for multirx pings, 50% more dofs...
#   but looking promising. down to 1900 so far... 11 iterations
#   down to 43 with 20 iterations
#   down to 10 with 40 iterations.

# with triplerx...
# 1st iteration: cost=321623.31
# 
best=fmin_powell(cost,vec_init,
                 ftol=1e-6,
                 xtol=1e-4,
                 disp=True,
                 maxfun=1e7,
                 maxiter=20,
                 callback=lambda vec: print(f"Min: cost={cost(vec):.2f}"))

##

# best_beacons=best


# For the beacon-to-beacon only setup, maybe this can be done as a matrix
# solve.
# each ping introduces an equation
# (t_ping_tx - (t_rx + shift_rx)) = dist(tx,rx)/c
# If x has t_ping_tx and shift_rx values...
# can solve lsq problem.

## 
best=fmin_powell(cost,best,
                 ftol=0.0001,
                 callback=lambda vec: print(f"Min: cost={cost(vec):.2f}"))

##

#from stompy.grid import unstructured_grid
from stompy.spatial import field
img=field.GdalGrid("../../model/grid/boundaries/junction-bathy-rgb.tif")

## 
# Plot a summary of solution:
shifts,ping_xyt=vec_to_params(best)
fig=plt.figure(3)
fig.clf()
fig.set_size_inches([10,8],forward=True)
fig,axs=plt.subplots(1,1,num=1)
ax=axs
ax.plot( rx_xy[:,0],rx_xy[:,1],'ro', label='Beacon',zorder=2)
    
img.plot(ax=ax,zorder=-2,alpha=0.5)

ax.plot( ping_xyt[:,0], ping_xyt[:,1],'k.',ms=3,label='Solutions',zorder=0)

sel3=(ds_strong.tag.values=='C2BA') & (np.isfinite(ds_strong.matrix).sum(axis=1)>2).values
ax.plot( ping_xyt[sel3,0], ping_xyt[sel3,1],'-',color='darkgreen',label='C2BA: 3-point solution',zorder=5)
sel2=(ds_strong.tag.values=='C2BA') 
ax.plot( ping_xyt[sel2,0], ping_xyt[sel2,1],'o',color='lightgreen',label='C2BA: 2-point non-unique',ms=5)
ax.legend(loc='upper left')
ax.axis( (647141.7878714591, 647354.6593804947, 4185819.2168760602, 4186039.6813729997) )
# Better, though there are a fair number of crazy positions.
# Ahh - but I'm including 2-rx solutions, which are poorly constrained

# How does C2BA compare to their data?

tekno=pd.read_csv('~/src/hor_flow_and_salmon/field/tags/2019/UTM/tag_C2BA_UTM.txt')
fig.tight_layout()
ax.plot( tekno[' X (East)'],
         tekno[' Y (North)'],
         color='orange',zorder=2,label='C2BA Tekno soln')
fig.savefig('c2ba-test-solutions.png')

##

# How do the ping intervals for C2BA look?
diffs=np.diff(ping_xyt[sel3,2])
diffs=diffs/np.round(diffs/5)
fig=plt.figure(2)
fig.clf()
sns.distplot(diffs,rug=True)
plt.xlabel('Inter-ping time (s)')
fig.savefig('c2ba-ping-times.png')


##

# # How persistent is the ping rate?  meh.  don't know if BB8A is moving around.
# # dt is clustered around 5.06
# # center 50% fall between 5.044s and 5.070s
# # range of 26ms
# df=ds.to_dataframe().reset_index()
# 
# df.groupby(['rx','tag']).size().sort_values()
# 
# sel=(df.tag=='BB8A') & (df.rx=='SM8')
# 
# tnums=df.tnum.values[sel]
# dt=np.diff(tnums)
# dt=dt[ dt<7.5 ]
# import seaborn as sns
# 
# plt.figure(1).clf()
# sns.distplot(dt,rug=True)

