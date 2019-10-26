# Explore and parse the raw data from the receivers
# Files in the AM1 folder:

#  33137152 Sep 16 12:10 am18-6002190531229.BIN
#       158 Sep 16 12:10 am18-6002190531229.CF2
#       244 Sep 16 12:10 am18-6002190531229.CFG
#  16262944 Sep 16 12:11 am18-6002190531229.D3D
#    794421 Sep 16 12:10 am18-6002190531229.DBG
#  51813494 Sep 16 12:12 am18-6002190531229.DET
#    492023 Sep 16 12:10 am18-6002190531229.EXT
#  39896932 Sep 16 12:12 am18-6002190531229.JST
#  28976197 Sep 16 12:13 am18-6002190531229.SEN
#  60681613 Sep 16 12:14 am18-6002190531229.SUM
#    194597 Sep 16 12:10 Downloaded 18_6002.docx
#

# BIN: binary, some strings.
# D3D: same, fewer strings
# DBG: timestamped log messages, seems to be hourly update messages.
# DET: now we're talking.  CSV.  looks like we get date,time, microseconds, maybe an epoch seconds
#   hex tag id.  5 numeric columns that are unclear.  maybe a pressure?   and  then a temperature.
#

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

def parse_tek(fn):
    df=pd.read_csv(fn,
                   names=['id','year','month','day','hour','minute','second','epoch','usec',
                          'tag','nbwQ','corrQ','num1','num2','one','pressure','temp'])

    dates=[ datetime.datetime(year=rec['year'],
                              month=rec['month'],
                              day=rec['day'],
                              hour=rec['hour'],
                              minute=rec['minute'],
                              second=rec['second'])
            for idx,rec in df.iterrows()]

    df['time'] = utils.to_dt64(np.array(dates)) + df['usec']*np.timedelta64(1,'us')

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
    
    return df3
## 

am1_full=parse_tek("2019 Data/HOR_Flow_TekDL_2019/AM1_186002/am18-6002190531229.DET")
sm3_full=parse_tek("2019 Data/HOR_Flow_TekDL_2019/SM3_187026/20190516/187026_190301_124447_P/sm18-7026190421300.DET")

##

# trim to common valid time period
valid_start=np.max( [am1_full.time.values[0],  sm3_full.time.values[0]] )
valid_end = np.min( [am1_full.time.values[-1], sm3_full.time.values[-1]] )

am1=am1_full[ (am1_full.time>=valid_start)&(am1_full.time<=valid_end) ].copy()
sm3=sm3_full[ (sm3_full.time>=valid_start)&(sm3_full.time<=valid_end) ].copy()

## 
for figi,df in enumerate([am1,sm3]):
    # Glance at some fields to see if they make sense
    plt.figure(1+figi).clf()
    fig,(axT,axP)=plt.subplots(2,1,sharex=True,num=1+figi)

    axT.plot(df.time,df.temp,label='temp')
    axT.legend(loc='upper right')
    axP.plot(df.time,df.pressure,label='pressure')
    axP.legend(loc='upper right')

## 
# When did am1 here sm3?

def remove_multipath(df):
    # Sometimes a single ping is heard twice (a local bounce)
    # when the same tag is heard in quick succession (<1s) drop the
    # second occurrence.
    delta_us=np.diff(df.time.values) / np.timedelta64(1,'us')
    assert np.all(delta_us>0)
    bounces=delta_us<1e6
    valid=np.r_[ True, delta_us>1e6 ]
    return df[valid].copy()

am1_to_am1=remove_multipath(am1[ am1.tag=='FF02' ])
am1_to_sm3=remove_multipath(sm3[ sm3.tag=='FF02' ])
sm3_to_sm3=remove_multipath(sm3[ sm3.tag=='FF26' ])
sm3_to_am1=remove_multipath(am1[ am1.tag=='FF26' ])

# Make sure everything is chronological
for df in [am1_to_am1,am1_to_sm3,sm3_to_sm3,sm3_to_am1]:
    assert np.diff(df.time.values).min()>np.timedelta64(0)

## 
# About 10k of each

t0=utils.to_dt64(max( am1_to_sm3.time.min(), sm3_to_am1.time.min() ))
tN=utils.to_dt64(min( am1_to_sm3.time.max(), sm3_to_am1.time.max() ))
t0N=[t0,tN]

def add_src_time(a_to_b,a_to_a,shift_us=0,shift_sec_per_day=0):
    """
    for each a_to_b, find the nearest in time a_to_a,
    and add a column for its time, named time_src
    """
    a_to_a_times=a_to_a.time.values + shift_us*np.timedelta64(1,'us')
    days=(a_to_a.time.values - t0)/np.timedelta64(86400,'s')
    a_to_a_times+= np.timedelta64(1,'us')*(1e6 * shift_sec_per_day * days).astype(np.int32)
    near=utils.nearest(a_to_a_times,
                       a_to_b.time.values)
    a_to_b['time_src']=a_to_a_times[near]
    trav_secs=(a_to_b.time - a_to_b.time_src)/np.timedelta64(1,'us') * 1e-6
    # some crazy numbers are throwing off the averages
    trav_secs[np.abs(trav_secs)>1000]=np.nan
    a_to_b['trav_secs']=trav_secs

#
# Look at the time series of trav_secs for each
#  Adding +-30secs to shift makes the time series "hairy"
#  which is probably that error I've been looking for where
#  the interval from the tags is not constant.  so this
#  should be getting the correct line up of pings.
#  it seems like am1 doesn't always do a good job of hearing itself?
shift=-1.5e6
shift_per_day=-0.315 # bingo.
add_src_time(am1_to_sm3,am1_to_am1,shift_us=shift,shift_sec_per_day=shift_per_day)
add_src_time(sm3_to_am1,sm3_to_sm3,shift_us=-shift,shift_sec_per_day=-shift_per_day)

a2s_good=am1_to_sm3[ np.abs(am1_to_sm3.trav_secs)<5 ]
s2a_good=sm3_to_am1[ np.abs(sm3_to_am1.trav_secs)<5 ]

plt.figure(13).clf()
plt.plot(am1_to_sm3.time,am1_to_sm3.trav_secs,alpha=0.1,label='am1 to sm3 all')
plt.plot(a2s_good.time,a2s_good.trav_secs,label='am1 to sm3 good')
plt.plot(sm3_to_am1.time,sm3_to_am1.trav_secs,alpha=0.1,label='sm3 to am1 all ')
plt.plot(s2a_good.time,s2a_good.trav_secs,label='sm3 to am1 good')
plt.legend()
plt.axis(ymin=-20,ymax=20)

##

# Not bad - so the pings in a2s_good and s2a_good do appear to be almost
# entirely good matches
# the drift, aside from clock updates, is down ~0.1s/month
# so if two pings are within 100 seconds of each other, how much drift occured?
#  4us.  So at this point we're no better than 4us.

a_matches=[]
# use indexes into the real list, not the pandas index.
for idx,(_,a2b_rec) in enumerate(a2s_good.iterrows()):
    match={}
    match['a2b_idx']=idx
    match['a2b_time_b']=a2b_rec['time']
    match['a2b_time_a']=a2b_rec['time_src']

    # Find a good s2a ping:
    b2a_idx=utils.nearest(s2a_good.time.values,a2b_rec.time.to_datetime64())
    b2a_rec=s2a_good.iloc[b2a_idx,:]
    match['b2a_idx']=b2a_idx
    match['b2a_time_b']=b2a_rec['time_src']
    match['b2a_time_a']=b2a_rec['time']
    match['match_diff']=a2b_rec['time']-b2a_rec['time']

    a_matches.append(match)

## 
a_df=pd.DataFrame(a_matches)

# a_df['b2a_time_a'] - a_df['b2a_time_b']
#   those are reasonable, showing that a is seeing the b2a ping 1s earlier than b.
# a_df['a2b_time_a'] - a_df['a2b_time_b']
#   these are also reasonable, also showing that a sees the a2b ping 1s earlier than b.
# good.

# almost the key calculation
# the difference in perceived duration from a's clock
a_df['mean_travel'] = -(
    (a_df['a2b_time_a'].values - a_df['b2a_time_a'].values) # roughly 1s
    -
    (a_df['a2b_time_b'].values - a_df['b2a_time_b'].values) 
)/2


##
from stompy.io.local import cdec
msd_flow=cdec.cdec_dataset('MSD',start_date=t0N[0],end_date=t0N[1],sensor=20,
                           cache_dir='.')
msd_stage=cdec.cdec_dataset('MSD',start_date=t0N[0],end_date=t0N[1],sensor=1,
                           cache_dir='.')

##

# Now to back out the asymmetry in travel

# how much ahead is b's clock, based on the a->b ping?
clock_err_a2b= a_df.a2b_time_b.values - (a_df.a2b_time_a.values + a_df.mean_travel.values )
# how much ahead is b's clock, based on the b->a ping?
clock_err_b2a= a_df.b2a_time_b.values - (a_df.b2a_time_a.values - a_df.mean_travel.values )

a_df['clock_skew']=(clock_err_a2b+clock_err_b2a)/2
# beware factors of 2 (i.e. work through whether this needs a factor of 2)
# also this is identical to zero.
a_df['travel_skew']=clock_err_a2b-clock_err_b2a

# boils down to we can't tell the difference between clock skew and travel skew.
# if clock skew can be modeled sufficiently, then we could make some headway.
# 

##

# - the mean_travel looks good. it's around 75ms, which is reasonable
# the variation is in good agreement with the variation in temperature.
# there is a bit of ghosting from (a) when a receiver doesn't hear its
# real pulse but does hear a bounce, and (b) some evidence of multipath
# between the two stations.

# as temperature increases, travel time decreases.
# this is the correct relation (https://www.engineeringtoolbox.com/sound-speed-water-d_598.html)
plt.figure(14).clf()
fig,axs=plt.subplots(3,1,num=14,sharex=True)

axs[0].plot( a_df['a2b_time_a'], a_df['mean_travel']/np.timedelta64(1,'us'), 'k.',alpha=0.2,ms=1 )
#axs[1].plot( msd_stage.time, msd_stage.sensor0001 )
axs[1].plot( msd_flow.time, msd_flow.sensor0020,label='Flow' )
#axs[1].plot(am1.time,am1.temp,label='Temp')
axs[0].set_ylabel('$\mu$s')
axs[0].axis(ymin=74e3,ymax=79e3)


#axs[2].plot( a_df['a2b_time_a'], a_df['travel_skew']/np.timedelta64(1,'us'), 'k.',alpha=0.2,ms=1 )
#axs[2].set_ylabel('Travel skew $\mu$s')

axs[2].plot( a_df['a2b_time_a'], a_df['clock_skew']/np.timedelta64(1,'us'), 'k.',alpha=0.2,ms=1 )
axs[2].set_ylabel('Clock skew $\mu$s')

# There is some diurnal-ish pattern in the clock skew - was optimistic that
# this would be related to flows, but there isn't actually that much daily
# change in flow, and it happens even when the flow is high and basically
# constant. maybe related to temperature, or internal clock adjustment.
# whatever it is, its amplitude is in the neighborhood of 3ms, so 60x
# greater than what we can expect to have 

##


bins=np.linspace(0,120,121)

# how well does AM1 hear itself?
plt.figure(14).clf()
fig,axs=plt.subplots(2,1,num=14)
axs[0].hist(1e-6*(np.diff(am1_to_am1.time)/np.timedelta64(1,'us')),
            bins=bins,log=True)
axs[1].hist(1e-6*(np.diff(sm3_to_sm3.time)/np.timedelta64(1,'us')),
            bins=bins,log=True)


##
shifts=np.arange(-1000e6,1000e6,1e6)
a_to_s_means=[]
s_to_a_means=[]

for shift in shifts:
    add_src_time(am1_to_sm3,am1_to_am1,shift_us=shift)
    add_src_time(sm3_to_am1,sm3_to_sm3,shift_us=-shift)

    # print("Shift us: %g  mean |am1_to_sm3|: %.3f  mean |sm3_to_am1|: %.3f"%
    #       (shift,
    #        np.abs(am1_to_sm3['trav_secs']).mean(),
    #        np.abs(sm3_to_am1['trav_secs']).mean()) )
    a_to_s_means.append( np.median(np.abs(am1_to_sm3['trav_secs']).mean()))
    s_to_a_means.append( np.median(np.abs(sm3_to_am1['trav_secs']).mean()))

# something is off - am1 to sm3 has an average offset something like 5--10
#   while sm3 to am1 has a typical offset of 40--45.
# changing from mean to median doesn't help [might have bungled that comparison]
# is it a clock drift issue? that they drift over time? [appears to be a drift issue.
#   maybe 0.3s/day].
# Also maybe am1 doesn't hear itself that well?

# How about around 2019-03-15
#  when did SM3 hear FF02 / AM1?
#   2019,03,15, 00,04,03  .708670    AM1 heard itself at 03:39, 04:09
#   2019,03,15, 00,11,37,
#   2019,03,15, 00,14,09,
#   2019,03,15, 00,14,39
#   2019,03,15, 00,21,13
#   2019,03,15, 00,27,16
# ...

# so in some cases as often as 30s apart, but misses a lot.
# and the intervals do not appear to be exactly 30s
# compare that to when AM1 heard itself
# this is basically every 30 seconds, though it sometimes hears
# one ping twice within a short period of time (~7ms)
    
##

# This is a sinusoid, roughly every 30 seconds
# so probably that means the they are sending a ping every 30 seconds.
# fine.
plt.figure(12).clf()
plt.plot(shifts/1e6,a_to_s_means,label='am1 to sm3')
plt.plot(shifts/1e6,s_to_a_means,label='sm3 to am1')
plt.legend()

##


    
## 

    
for shift in np.linspace(-60e6,60e6,241):
    add_src_time(am1_to_sm3,am1_to_am1,shift_us=shift)
    add_src_time(sm3_to_am1,sm3_to_sm3,shift_us=-shift)

    # There might be some clock skew, but assume that anything more than 30 seconds
    # is definitely not the same ping.

    # raw travel times
    plt.figure(10).clf()

    def trim(x):
        bad=np.abs(x)>30.0
        return x[ ~bad ]

    plt.hist([trim(am1_to_sm3.trav_secs.values),
              trim(sm3_to_am1.trav_secs.values)],bins=100)
    plt.draw()
    plt.pause(0.01)
    #plt.plot( am1_to_sm3.time,
    #          am1_to_sm3.time_src,
    #          'g.')
    # 1:1
    #t0N=am1_to_sm3.time.values[ [0,-1] ]
    #plt.plot(t0N,t0N,'k-',lw=0.5)

##

# Broad strokes - compare pressure time series:
am1.name='am1'
sm3.name='sm3'

plt.figure(11).clf()
fig,(axT,axP)=plt.subplots(2,1,sharex=True,num=11)

from stompy import filters

def lp(x):
    return filters.lowpass_fir(x,winsize=50)


# upper bound is 1200s
# lower bound is -500
shift=np.timedelta64(0,'s')
for df in am1,sm3:
    axT.plot(df.time+shift,lp(df.temp.values),label='temp '+df.name)
    axP.plot(df.time+shift,lp( (df.pressure - df.pressure.mean()).values),label='pressure '+df.name)
    shift=np.timedelta64(-500,'s')
    
axT.legend(loc='upper right')
axP.legend(loc='upper right')
axP.axis( (737161.400148251, 737167.0008022288, -6908.232726575427, -1089.3113427372318))

##

t0=utils.to_dnum(am1.time)[0]

lag_press=utils.find_lag( utils.to_dnum(am1.time)-t0, lp(am1.pressure.values),
                          utils.to_dnum(sm3.time)-t0,lp(sm3.pressure.values) )

lag_temp=utils.find_lag( utils.to_dnum(am1.time)-t0,lp(am1.temp.values),
                         utils.to_dnum(sm3.time)-t0,lp(sm3.temp.values) )

print("Pressure lag:    %gs"%(86400*lag_press))
print("Temperature lag: %gs"%(86400*lag_temp))



##

# what kind of change in travel time is expected?
L=113.0 # m
c=1500 # m/s
u=0.5 # m/s

trav=L/c # ~75 ms
trav_pos=L/(c+u)
trav_neg=L/(c-u)

print(f"Expected travel time: {trav*1000:.5f} ms")
print(f"  Expected asymmetry: {1000*(trav_pos-trav_neg):.5f}ms")
# 50us.
