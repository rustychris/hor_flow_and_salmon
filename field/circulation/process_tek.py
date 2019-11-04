"""
Expand on the code in parse_tek.
and jettison some of the exploratory ground-truthing
code
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from stompy import utils

# tag to station:
# FF02: AM1
# FF26: SM3
# distance between those is 113m.

##

def parse_tek(det_fn,cf2_fn=None):
    if cf2_fn is None:
        fn=det_fn.replace('.DET','.CF2')
        if os.path.exists(fn):
            cf2_fn=fn

    df=pd.read_csv(det_fn,
                   names=['id','year','month','day','hour','minute','second','epoch','usec',
                          'tag','nbwQ','corrQ','num1','num2','one','pressure','temp'])

    if 0:
        # this way is slow, and I *think* yields PST times, where epoch is UTC.
        dates=[ datetime.datetime(year=rec['year'],
                                  month=rec['month'],
                                  day=rec['day'],
                                  hour=rec['hour'],
                                  minute=rec['minute'],
                                  second=rec['second'])
                for idx,rec in df.iterrows()]
        df['time'] = utils.to_dt64(np.array(dates)) + df['usec']*np.timedelta64(1,'us')
    else:
        # this is quite fast and should yield UTC.
        # the conversion in utils happens to be good down to microseconds, so we can
        # do this in one go
        df['time'] = utils.unix_to_dt64(df['epoch'] + df['usec']*1e-6)
        
    # clean up time:
    bad_time= (df.time<np.datetime64('2018-01-01'))|(df.time>np.datetime64('2022-01-01'))
    df2=df[~bad_time].copy()

    # clean up temperature:
    df2.loc[ df.temp<-5, 'temp'] =np.nan
    df2.loc[ df.temp>35, 'temp'] =np.nan

    # clean up pressure
    df2.loc[ df2.pressure>160e3, 'pressure']=np.nan
    df2.loc[ df2.pressure<110e3, 'pressure']=np.nan

    # trim to first/last valid pressure
    valid_idx=np.nonzero( np.isfinite(df2.pressure.values) )[0]
    df3=df2.iloc[ valid_idx[0]:valid_idx[-1]+1, : ].copy()

    df3['tag']=[ s.strip() for s in df3.tag.values ]

    ds=xr.Dataset.from_dataframe(df3)

    if cf2_fn is not None:
        cf=pd.read_csv(cf2_fn,header=None)
        local_tag=cf.iloc[0,1].strip()
        ds['beacon_id']=local_tag

    return ds

## 

# what file would tell us the tag id for the colocated beacon?
# second column of CF2 file?
# this should be ff02:
am1_full=parse_tek("2019 Data/HOR_Flow_TekDL_2019/AM1_186002/am18-6002190531229.DET")
# and this ff26
sm3_full=parse_tek("2019 Data/HOR_Flow_TekDL_2019/SM3_187026/20190516/187026_190301_124447_P/sm18-7026190421300.DET")

##

A_full=am1_full
B_full=sm3_full

# trim to common valid time period
valid_start=np.max( [A_full.time.values[0],  B_full.time.values[0]] )
valid_end = np.min( [A_full.time.values[-1], B_full.time.values[-1]] )

A=A_full.isel( index=(A_full.time>=valid_start)&(A_full.time<=valid_end) ).copy()
B=B_full.isel( index=(B_full.time>=valid_start)&(B_full.time<=valid_end) ).copy()

## 
for figi,df in enumerate([A,B]):
    # Glance at some fields to see if they make sense
    plt.figure(1+figi).clf()
    fig,(axT,axP)=plt.subplots(2,1,sharex=True,num=1+figi)

    axT.plot(df.time,df.temp,label='temp')
    axT.legend(loc='upper right')
    axP.plot(df.time,df.pressure,label='pressure')
    axP.legend(loc='upper right')

## 
# When did A here B?

def remove_multipath(ds):
    # Sometimes a single ping is heard twice (a local bounce)
    # when the same tag is heard in quick succession (<1s) drop the
    # second occurrence.
    delta_us=np.diff(ds.time.values) / np.timedelta64(1,'us')
    assert np.all(delta_us>0)
    bounces=delta_us<1e6
    valid=np.r_[ True, delta_us>1e6 ]
    return ds.isel(index=valid).copy()

def x_to_y(x,y):
    """
    x,y: datasets of tag detections.
    returns the subset of detections in y that are coming from
    x, and not obviously multipaths
    """
    transits=y.isel( index=(y.tag==x.beacon_id) )
    good_transits=remove_multipath(transits)
    return good_transits

A_to_A=x_to_y(A,A)
A_to_B=x_to_y(A,B)
B_to_B=x_to_y(B,B)
B_to_A=x_to_y(B,A)

# Make sure everything is chronological
for ds in [A_to_A,A_to_B,B_to_B,B_to_A]:
    assert np.diff(ds.time.values).min()>np.timedelta64(0)

## 

# prescribe these so they're not changing with each pair
t0=np.datetime64('2019-03-01T00:00')
tN=np.datetime64('2019-05-08T00:00')

#t0=utils.to_dt64(max( A_to_B.time.values.min(), B_to_A.time.values.min() ))
#tN=utils.to_dt64(min( A_to_B.time.values.max(), B_to_A.time.values.max() ))
t0N=[t0,tN]

## 

from stompy.io.local import cdec
msd_flow=cdec.cdec_dataset('MSD',start_date=t0N[0],end_date=t0N[1],sensor=20,
                           cache_dir='.')
msd_stage=cdec.cdec_dataset('MSD',start_date=t0N[0],end_date=t0N[1],sensor=1,
                           cache_dir='.')
##

def add_src_time(a_to_b,a_to_a,shift_us=0,drift_us_per_day=0):
    """
    for each a_to_b, find the nearest in time a_to_a,
    and add a column for its time, named time_src.
    shift_us: constant time offset
    drift_us_per_day: linear drift, starting at 0 at t0 (defined
    just above), in microseconds per day.
    
    originally this would record the src time 
    """
    days=(a_to_a.time.values - t0)/np.timedelta64(86400,'s')
    shift=(shift_us + drift_us_per_day * days).astype(np.int64)
    a_to_a_times=a_to_a.time.values
    a_to_a_times_shifted=a_to_a_times + shift*np.timedelta64(1,'us')
    near=utils.nearest(a_to_a_times_shifted,
                       a_to_b.time.values)
    # choice here whether to use the shifted or unshifted time.
    # now trying the unshifted
    a_to_b['time_src']=('index',),a_to_a_times[near]
    # but save the shifted version, too
    a_to_b['time_src_shifted']=('index',),a_to_a_times_shifted[near]

    # and calculate trav_seconds with the shifted data, because it
    # is used to weed out bad matches
    trav_secs=(a_to_b.time - a_to_b.time_src_shifted)/np.timedelta64(1,'us') * 1e-6
    # some crazy numbers are throwing off the averages
    trav_secs[np.abs(trav_secs)>1000]=np.nan
    # note that this can be negative
    a_to_b['trav_secs']=('index',),trav_secs

# Look at the time series of trav_secs for each
#  Adding +-30secs to shift makes the time series "hairy"
#  which is probably that error I've been looking for where
#  the interval from the tags is not constant.  so this
#  should be getting the correct line up of pings.
#  it seems like A doesn't always do a good job of hearing itself?
#

# Since ultimately we want to compute lags between multiple pairs,
# better to use the shift and drift only to match pings, but
# compute the lags with the real clocks.
# I think..

shift=-1.5e6 # HERE HERE will have to figure out how to deal with this
# less manually
drift_us_per_day=-315000 # bingo.
add_src_time(A_to_B,A_to_A,shift_us=shift,drift_us_per_day=drift_us_per_day)
add_src_time(B_to_A,B_to_B,shift_us=-shift,drift_us_per_day=-drift_us_per_day)

## 
a2b_good=A_to_B.isel(index=np.abs(A_to_B.trav_secs.values)<5)
b2a_good=B_to_A.isel(index=np.abs(B_to_A.trav_secs.values)<5)

plt.figure(13).clf()
plt.plot(A_to_B.time,A_to_B.trav_secs,alpha=0.1,label='A to B all')
plt.plot(a2b_good.time,a2b_good.trav_secs,label='A to B good')
plt.plot(B_to_A.time,B_to_A.trav_secs,alpha=0.1,label='B to A all ')
plt.plot(b2a_good.time,b2a_good.trav_secs,label='B to A good')
plt.legend()
plt.axis(ymin=-20,ymax=20)

##

# The old code is below, where I try to match pairs of a->b and b->a
# pings.
# instead, could treat each as a timeseries.
# general idea being that we want to interpolate over the
# interval between individual pings, such that a linear drift
# can be exactly accounted for.

# the two quantities I want are mean_travel and clock_skew.

mean_travel = 0.5 * ( a2b_time_b - a2b_time_a  + b2a_time_a - b2a_time_b)
clock_skew= 0.5*(a2b_time_b+b2a_time_b) - 0.5*(a2b_time_a + b2a_time_a)

##

# rewrite to group terms by ping
#mean_travel = 0.5*(a2b_time_b - a2b_time_a) + 0.5*(b2a_time_a - b2a_time_b)
#clock_skew  = 0.5*(a2b_time_b - a2b_time_a) - 0.5*(b2a_time_a - b2a_time_b)

all_times=np.unique( np.concatenate( [a2b_good.time.values,
                                      b2a_good.time.values] ))

a2b_transit_us=np.interp( utils.to_dnum(all_times),
                          utils.to_dnum(a2b_good.time.values),
                          (a2b_good.time.values - a2b_good.time_src.values)/np.timedelta64(1,'us'))

b2a_transit_us=np.interp( utils.to_dnum(all_times),
                          utils.to_dnum(b2a_good.time.values),
                          (b2a_good.time.values - b2a_good.time_src.values)/np.timedelta64(1,'us'))

mean_travel_us=(a2b_transit_us+b2a_transit_us)/2
clock_skew_us=(a2b_transit_us-b2a_transit_us)/2

##

a2b_times=a2b_good.time.values
b2a_times=b2a_good.time.values

# Not bad - so the pings in a2s_good and s2a_good do appear to be almost
# entirely good matches
# the drift, aside from clock updates, is down ~0.1s/month
# so if two pings are within 100 seconds of each other, how much drift occured?
#  4us.  So at this point we're no better than 4us.

# again, here we want to match pings based on the
# shifted time, but [maybe] record the differences
# using the real instrument time.
a_matches=[]

a2b_times=a2b_good.time.values
b2a_times=b2a_good.time.values
if 0:
    #old code, when time_src included the shift
    a2b_time_srcs=a2b_good.time_src.values
    b2a_time_srcs=b2a_good.time_src.values
else:
    # new code, diagnosing effect of time_src vs time_src_shifted
    # using time_src_shifted gives much tighter results in fig 14.
    # I think that's because the match difference doesn't
    # contaminate the lags as much when the times are already
    # shifted.
    
    # this choice does not affect the choice of ping pairs,
    # as that choice is made solely on destination times.
    a2b_time_srcs=a2b_good.time_src_shifted.values
    b2a_time_srcs=b2a_good.time_src_shifted.values

for a2b_idx in range(len(a2b_good.index)):
    match={}
    match['a2b_idx']=a2b_idx
    match['a2b_time_b']=a2b_times[a2b_idx]
    match['a2b_time_a']=a2b_time_srcs[a2b_idx]

    # Find a good b2a ping:
    b2a_idx=utils.nearest(b2a_times,match['a2b_time_b'])
    match['b2a_idx']=b2a_idx
    match['b2a_time_b']=b2a_time_srcs[b2a_idx]
    match['b2a_time_a']=b2a_times[b2a_idx]
    match['match_diff']=match['a2b_time_b'] - match['b2a_time_a']
    a_matches.append(match)

## 
a_df=pd.DataFrame(a_matches)

# almost the key calculation
a_df['mean_travel'] =  0.5 * ( a_df.a2b_time_b - a_df.a2b_time_a + a_df.b2a_time_a - a_df.b2a_time_b)

# This is what I want: 
#a_df['clock_skew']=0.5*(a_df.a2b_time_b+a_df.b2a_time_b) - 0.5*(a_df.a2b_time_a + a_df.b2a_time_a)
# But date arithmetic requires it be done in a different order:
a_df['clock_skew']=0.5*(  a_df.a2b_time_b - a_df.a2b_time_a + a_df.b2a_time_b - a_df.b2a_time_a)


# - the mean_travel looks good. it's around 75ms, which is reasonable
# the variation is in good agreement with the variation in temperature.
# there is a bit of ghosting from (a) when a receiver doesn't hear its
# real pulse but does hear a bounce, and (b) some evidence of multipath
# between the two stations.

# as temperature increases, travel time decreases.
# this is the correct relation (https://www.engineeringtoolbox.com/sound-speed-water-d_598.html)
plt.figure(14).clf()
fig,axs=plt.subplots(2,1,num=14,sharex=True)

#axs[0].plot( a_df['a2b_time_a'], a_df['mean_travel']/np.timedelta64(1,'us'), 'k.',alpha=0.2,ms=1 ,
#             label='Mean travel time')

# scatter with the match difference as color. idea being that when this is calculated without
# shifts, it gets fuzzier, but maybe that's because the mean travel time is thrown off by
# the drift, and the difference in match times causes that effect to be noisier.
scat=axs[0].scatter( a_df['a2b_time_a'], a_df['mean_travel']/np.timedelta64(1,'us'),
                     1, 1e-6*(a_df['match_diff']/np.timedelta64(1,'us')),
                     alpha=0.2,cmap='jet')
scat.set_clim([-300,300])

axs[0].plot(all_times, mean_travel_us,'k.',alpha=0.2,ms=1,label='Mean travel time')

#axs[1].plot(A.time,A.temp,label='Temp')
axs[0].set_ylabel('$\mu$s')
axs[0].axis(ymin=74e3,ymax=79e3)
#axs[0].legend(loc='upper right')

plt.colorbar(scat,ax=axs[0])

axs[1].plot( a_df['a2b_time_a'], 1e-6*(a_df['clock_skew']/np.timedelta64(1,'us')), 'k.',alpha=0.2,ms=1,
             label='clock skew')

axs[1].plot( all_times, clock_skew_us, 'g.',alpha=0.2,ms=1,
             label='clock skew (cont.)')

axs[1].set_ylabel('Clock skew (s)')
axs[1].legend(loc='upper right')

