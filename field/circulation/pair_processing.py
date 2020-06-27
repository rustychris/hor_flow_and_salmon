"""
Expand on the code in parse_tek.
and jettison some of the exploratory ground-truthing
code
"""

import pandas as pd
import numpy as np
import datetime
import logging as log
from stompy import filters
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import fmin

from stompy import utils

# tag to station:
# FF02: AM1
# FF26: SM3
# distance between those is 113m.

##

from parse_tek import parse_tek, remove_multipath

## 

def x_to_y(x,y):
    """
    x,y: datasets of tag detections.
    returns the subset of detections in y that are coming from
    x, and not obviously multipaths
    """
    transits=y.isel( index=(y.tag==x.beacon_id) )
    good_transits=remove_multipath(transits)
    return good_transits

def add_src_time(a_to_b,a_to_a,shift_us=0,drift_us_per_day=0,t0=np.datetime64("1970-01-01")):
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

class BeaconPair(object):
    """
    Process a pair of receivers/beacons, generating a time series of
    mean travel time and clock skew.
    """
    # prescribe these so they're not changing with each pair
    t0=np.datetime64('2019-03-01T00:00')
    tN=np.datetime64('2019-05-08T00:00')
    
    shift_search=np.linspace(-5e6,5e6,21)
    drift_search=np.linspace(-1e6,1e6,21)

    clip_start=None
    clip_stop=None
    
    def __init__(self,fnA,fnB,label="n/a",**kw):
        self.fnA=fnA
        self.fnB=fnB
        self.label=label

        utils.set_keywords(self,kw)
        
        self.parse_data()
        self.trim_to_common()
        self.get_pings()
        self.tune_shifts()
        self.select_good_transits()
        self.figure_matches()
        self.calc_travel_and_skew()
        self.figure_transit_and_skew()
        
    def parse_data(self):
        self.A_full=parse_tek(self.fnA)
        self.B_full=parse_tek(self.fnB)

    def trim_to_common(self):
        """
        trim to common valid time period, including optional clip
        """
        valid_start=np.max( [self.A_full.time.values[0],  self.B_full.time.values[0]] )
        valid_end = np.min( [self.A_full.time.values[-1], self.B_full.time.values[-1]] )

        if (self.clip_start is not None) and (self.clip_start>valid_start):
            valid_start=self.clip_start
        if (self.clip_stop is not None) and (self.clip_stop<valid_end):
            valid_end=self.clip_end

        self.A=self.A_full.isel( index=(self.A_full.time>=valid_start)&(self.A_full.time<=valid_end) ).copy()
        self.B=self.B_full.isel( index=(self.B_full.time>=valid_start)&(self.B_full.time<=valid_end) ).copy()
        
    def figure_environmental(self):
        for figi,df in enumerate([self.A,self.B]):
            # Glance at some fields to see if they make sense
            plt.figure(1+figi).clf()
            fig,(axT,axP)=plt.subplots(2,1,sharex=True,num=1+figi)

            axT.plot(df.time,df.temp,label='temp')
            axT.legend(loc='upper right')
            axP.plot(df.time,df.pressure,label='pressure')
            axP.legend(loc='upper right')

    def get_pings(self):
        self.A_to_A=x_to_y(self.A,self.A)
        self.A_to_B=x_to_y(self.A,self.B)
        self.B_to_B=x_to_y(self.B,self.B)
        self.B_to_A=x_to_y(self.B,self.A)

    def tune_shifts(self):
        # Since ultimately we want to compute lags between multiple pairs,
        # better to use the shift and drift only to match pings, but
        # compute the lags with the real clocks.
        # I think..

        def cost(params):
            shift,drift=params
            add_src_time(self.A_to_B,self.A_to_A,shift_us=shift,drift_us_per_day=drift,t0=self.t0)
            add_src_time(self.B_to_A,self.B_to_B,shift_us=-shift,drift_us_per_day=-drift,t0=self.t0)

            # Try to develop a metric for tuning the shifts.
            nan_count=( np.sum(np.isnan(self.A_to_B.trav_secs) ) +
                        np.sum( np.isnan(self.B_to_A.trav_secs) ) )
            if nan_count:
                print("%d nan travel times"%nan_count)
            scoreAB=np.sqrt( np.nanmean((self.A_to_B.trav_secs.values)**2) )
            scoreBA=np.sqrt( np.nanmean((self.B_to_A.trav_secs.values)**2) )
            return scoreAB + scoreBA
        self.cost=cost

        # grid search to start with:
        shifts=self.shift_search
        drifts=self.drift_search

        best_c=np.inf
        best_shift=None
        best_drift=None
        for shift in self.shift_search:
            for drift in self.drift_search:
                c=cost([shift,drift])
                if c<best_c:
                    best_shift=shift
                    best_drift=drift
                    best_c=c
                    print(f"shift: {shift} drift: {drift} cost: {c}")

        if not np.isfinite(best_c):
            print("All costs were bad??")

        # Refine that
        best=fmin(cost,[best_shift,best_drift])
        # be sure the last call uses the best values
        cost(best)
        self.shift=best_shift
        self.drift=best_drift

    def select_good_transits(self):
        self.a2b_good=self.A_to_B.isel(index=np.abs(self.A_to_B.trav_secs.values)<5)
        self.b2a_good=self.B_to_A.isel(index=np.abs(self.B_to_A.trav_secs.values)<5)

    def figure_matches(self):
        plt.figure(13).clf()
        plt.plot(self.A_to_B.time,self.A_to_B.trav_secs,alpha=0.1,label='A to B all')
        plt.plot(self.a2b_good.time,self.a2b_good.trav_secs,label='A to B good')
        plt.plot(self.B_to_A.time,self.B_to_A.trav_secs,alpha=0.1,label='B to A all ')
        plt.plot(self.b2a_good.time,self.b2a_good.trav_secs,label='B to A good')
        plt.legend(loc='upper left')
        plt.axis(ymin=-20,ymax=20)

    def calc_travel_and_skew(self):
        # the two quantities I want are mean_travel and clock_skew.
        # mean_travel = 0.5 * ( a2b_time_b - a2b_time_a  + b2a_time_a - b2a_time_b)
        # clock_skew= 0.5*(a2b_time_b+b2a_time_b) - 0.5*(a2b_time_a + b2a_time_a)
        
        # rewrite to group terms by ping
        #mean_travel = 0.5*(a2b_time_b - a2b_time_a) + 0.5*(b2a_time_a - b2a_time_b)
        #clock_skew  = 0.5*(a2b_time_b - a2b_time_a) - 0.5*(b2a_time_a - b2a_time_b)

        all_times=np.unique( np.concatenate( [self.a2b_good.time.values,
                                              self.b2a_good.time.values] ))

        if len(self.a2b_good.time.values)==0:
            log.error("No a2b matches were good")
        else:
            a2b_transit_us=np.interp( utils.to_dnum(all_times),
                                      utils.to_dnum(self.a2b_good.time.values),
                                      (self.a2b_good.time.values - self.a2b_good.time_src.values)/np.timedelta64(1,'us'))

        if len(self.b2a_good.time.values)==0:
            log.error("No b2a matches were good")
        else:
            b2a_transit_us=np.interp( utils.to_dnum(all_times),
                                      utils.to_dnum(self.b2a_good.time.values),
                                      (self.b2a_good.time.values - self.b2a_good.time_src.values)/np.timedelta64(1,'us'))

        if len(all_times)==0:
            log.error("Nothing good")
            self.transits=None
            return
        
        mean_travel_us=(a2b_transit_us+b2a_transit_us)/2
        # when skew is positive, that means it appears that a2b takes longe than b2a
        # that would be the case when a is ahead of b.
        # e.g. if a is 1 second ahead of b, and transit time is 1 second, then
        # a2b transit is 2 seconds, b2a transit is 0
        clock_skew_us=(a2b_transit_us-b2a_transit_us)/2

        self.transits=xr.Dataset()
        self.transits['time']=('time',),all_times
        self.transits['mean_travel_us']=('time',),mean_travel_us
        self.transits['clock_skew_us']=('time',),clock_skew_us
        self.transits['clock_skew_us'].attrs['desc']="A clock leads B clock by this many usecs"


        # estimating how stale each measurement is
        t_all_us=(self.transits.time.values-self.t0)/np.timedelta64(1,'us')
        t_Asrc_us=(self.a2b_good.time.values-self.t0)/np.timedelta64(1,'us')
        t_Bsrc_us=(self.b2a_good.time.values-self.t0)/np.timedelta64(1,'us')
        deltasA=t_all_us - utils.nearest_val(t_Asrc_us,t_all_us)
        deltasB=t_all_us - utils.nearest_val(t_Bsrc_us,t_all_us)
        stale_time_us=np.abs(deltasA) + np.abs(deltasB)
        self.transits['stale_us']=('time',),stale_time_us

        # true when A was the source for this time, false when B was the source
        src_is_A=np.zeros(len(all_times))
        src_is_A[ np.searchsorted(all_times,self.a2b_good.time.values) ]=1
        self.transits['src_is_A']=('time',), src_is_A

    def figure_transit_and_skew(self,color_by_source=True):
        if self.transits is None:
            print("No transits. No plot.")
            return
        plt.figure(14).clf()
        fig,axs=plt.subplots(2,1,num=14,sharex=True)

        if color_by_source:
            selA=self.transits.src_is_A.values==1
            selB=~selA
            for sel,col in [ (selA,'k'),(~selA,'r')]:
                axs[0].plot(self.transits.time[sel],
                            self.transits.mean_travel_us[sel] * 1e-3, 
                            col+'.',alpha=0.2,ms=1,label='Mean travel time')
        else:
            axs[0].plot(self.transits.time,self.transits.mean_travel_us * 1e-3, 
                        'k.',alpha=0.2,ms=1,label='Mean travel time')
        axs[0].set_ylabel('(ms)')
        # axs[0].axis(ymin=74,ymax=79)
        lims=np.percentile(1e-3*self.transits.mean_travel_us.values,[25,75])
        width=lims[1]-lims[0]
        axs[0].axis(ymin=lims[0]-width,ymax=lims[1]+width)
        
        axs[0].legend(loc='upper right')

        axs[1].plot( self.transits.time,self.transits.clock_skew_us * 1e-6,
                     'g.',alpha=0.2,ms=1,
                     label='Clock skew (cont.)')
        axs[1].legend(loc='upper right')

        
#pair=BeaconPair("2019 Data/HOR_Flow_TekDL_2019/AM1_186002/am18-6002190531229.DET",
#                "2019 Data/HOR_Flow_TekDL_2019/SM3_187026/20190516/187026_190301_124447_P/sm18-7026190421300.DET")

##

AM1="2019 Data/HOR_Flow_TekDL_2019/AM1_186002/am18-6002190531229.DET"
AM3="2019 Data/HOR_Flow_TekDL_2019/AM3_186003/20190517/186003_190217_125944_P/186003_190217_125944_P.DET"
AM4="2019 Data/HOR_Flow_TekDL_2019/AM4_186008/20190517/186008_190517_130946_P/186008_190517_130946_P.DET"

AM7="2019 Data/HOR_Flow_TekDL_2019/AM7_186007/20190517/186007_190217_135133_P/am18-6007190481351.DET"
AM9="2019 Data/HOR_Flow_TekDL_2019/AM9_186004/20190517/186004_190311_144244_P/am18-6004190440924.DET"

SM1="2019 Data/HOR_Flow_TekDL_2019/SM1_187023/20190517/187023_190416_050506_P/sm18-7023190711049.DET"
SM2="2019 Data/HOR_Flow_TekDL_2019/SM2_187013/2018_7013/187013_190516_143749_P.DET"
SM3="2019 Data/HOR_Flow_TekDL_2019/SM3_187026/20190516/187026_190301_124447_P/sm18-7026190421300.DET"
SM8="2019 Data/HOR_Flow_TekDL_2019/SM8_187028/20190516/187028_190216_175239_P/sm18-7028190421324.DET"
SM7="2019 Data/HOR_Flow_TekDL_2019/SM7_187017/20190516/187017_190225_154939_P/sm18-7017190391457.DET"

##
# original test pair
# seems okay. cost down to 6.8
AM1_SM3=BeaconPair(AM1,SM3,"AM1_SM3")

##
# good..
# cost down to 2.2.  Seems like more variation in the travel time.
# almost a range of 9ms?
# seems that this one is driving the errors in circulation.
# the reverse (below) gets the same result.
# the path here is a bit long, and maybe over some shallow water?
# looks like there is a 10:1 imbalance in who heard whom.
# most of the crazy signal is during long periods when there are no
# pings from AM3 to SM8.
AM3_SM8=BeaconPair(AM3,SM8,"AM3_SM8")

##

# okay - lots of multipath, it seems. sometimes two echoes, and they can
# be stronger than the straight path.
# cost down to 3.0
SM8_AM4=BeaconPair(SM8,AM4,"SM8_AM4")

##

# cost down to 0.74.
# some multipath
AM4_AM3=BeaconPair(AM4,AM3,"AM4_AM3")

##

# some issues with one or more of these beacons.
# just much deeper.
# but it looks like maybe it wiggled a lot.
# the travel time has crazy noise, like +-0.3ms
# which is 40cm or so. decorrelation time of maybe
# 10-20s?
# cost gets down to 1.22
# Mike confirmed that AM9 is on a short tether.
# potentially can get better information by including
# the tilt.
AM7_AM9=BeaconPair(AM7,AM9,"AM7_AM9")

##

# and these never converge on a good clock drift.
# maybe outside the checked range.
# This one looks like the clock had 3 distinct drifts
# Try fitting just the last of those, which is the longest.
# with original settings, it gets down to cost of 1.80
# seems that the "real" shift is about -15s.
# expanding the search range to include that
# gets the cost a bit lower, and lets travel time
# have a reasonable 32 ms average.
AM9_SM7=BeaconPair(AM9,SM7,"AM9_SM7",clip_start=np.datetime64("2019-04-15"),
                   shift_search=np.linspace(-20e6,-10e6,11) )

##

# This pair has a few pings with large offsets, so some trav_secs
# are nan.  If those are ignored, it works okay, though ending
# function value is 14.8
SM7_AM7=BeaconPair(SM7,AM7,"SM7_AM7",
                   shift_search=np.linspace(-20e6,-20e6,11) )

AM1_SM8=BeaconPair(AM1,SM8,"AM1_SM8") # only 22 pings either way.

# How does this do compared to the reverse?  same.
# the reverse is the main source of circulation error, and this looks no
# better
SM8_AM3=BeaconPair(SM8,AM3,"SM8_AM3")

# Fair bit of multipath and some just plain noise.
# but probably an okay line-up
AM3_AM1=BeaconPair(AM3,AM1)

##

# SM1 clock jumps back 9 hours on
# 2019-03-28, from 0900 to 0000.
# There are several suspect jumps
# in time. Try trimming to the last portion
# of the data.
# next problem is that SM2 doesn't hear itself.
# supposedly it's FF13, but FF05 is what it hears
# most often and FF20, FF26, FF28 after that.
# FF05 is AM6
# FF20 is SM4.  those are both downstream a good ways
# FF26 is SM3
# FF28 is SM8.
# All of that seems to point to the file I think is SM2 actually
# being AM8.
SM1_SM2=BeaconPair(SM1,SM2,"SM1_SM2",clip_start=np.datetime64("2019-03-28 12:00"))

##

# This trio doesn't give good circulation because
# AM9 is on a tether and imparts a lot of noise in the
# transit time.
legs=[AM7_AM9,
      AM9_SM7,
      SM7_AM7]


##

# What does the combination of all 3 tell us?
# these legs form a CW triangle, and AM4-AM3
# are closer to shore river-right. there should be
# a net CW circulation. I get a decent signal at
# -3.3us
legs=[AM3_SM8,
      SM8_AM4,
      AM4_AM3]

##

# This triangle is no good because AM1_SM8 has basically no
# data
legs=[AM1_SM8,
      SM8_AM3,
      AM3_AM1]

##

legs=[SM1_SM2,
      SM2_AM1,
      AM1_SM1]

##

t_complete=np.unique( np.concatenate( [ leg.transits.time.values for leg in legs] ) )

skews=[ np.interp( utils.to_dnum(t_complete),
                   utils.to_dnum(leg.transits.time.values), leg.transits.clock_skew_us.values,
                   left=np.nan,right=np.nan)
        for leg in legs]

travels=[ np.interp( utils.to_dnum(t_complete),
                     utils.to_dnum(leg.transits.time.values), leg.transits.mean_travel_us.values,
                     left=np.nan,right=np.nan)
          for leg in legs]

stales=[ np.interp(utils.to_dnum(t_complete),
                   utils.to_dnum(leg.transits.time.values), leg.transits.stale_us.values,
                   left=np.nan,right=np.nan)
        for leg in legs]


##

from stompy.io.local import cdec
msd_flow=cdec.cdec_dataset('MSD',start_date=BeaconPair.t0,end_date=BeaconPair.tN,sensor=20,
                           cache_dir='.')
msd_stage=cdec.cdec_dataset('MSD',start_date=BeaconPair.t0,end_date=BeaconPair.tN,sensor=1,
                           cache_dir='.')

##


net_skew=skews[0]+skews[1]+skews[2]
net_travel=travels[0]+travels[1]+travels[2]
net_stale=stales[0]+stales[1]+stales[2]


plt.figure(19).clf()
from matplotlib.gridspec import GridSpec

fig=plt.figure(num=19)
#fig,axs=plt.subplots(3,1,sharex=True,num=19)
gs=GridSpec(3,4)
axs=[fig.add_subplot(gs[0,:-1])]
axs.append( fig.add_subplot(gs[1,:-1],sharex=axs[0]) )
axs.append( fig.add_subplot(gs[2,:-1],sharex=axs[1]) )

select=net_stale<250e6
axs[0].plot(t_complete,net_skew,alpha=0.2,label='Net skew~circulation')
axs[0].plot(t_complete[select],net_skew[select],'g.',label='Low stale')
axs[0].set_ylabel(r"Circ. $\mu$s")


# With the low-staleness samples, I do get some periods, like
# 2019-04-15, where there is a long series of good samples, and
# the signal hovers, noisily, around -5us.
# does seem that most places where there is a stretch of good samples,
# it tends to be close to that value.

# what value might I expect? sign would I expect here?
# cycle_time_us=np.nanmean(net_travel) # 174ms

# say half the cycle is in 0.25m/s flow
# and the other half is 0.75m/s flow
# speed of sound is about 1500m/s
# that means in that in on direction the ping is fighting
# on average a 0.25m/s "headwind"
# and in the other direction is has a 0.25m/s "tailwind"
# need pencil and paper to do this exactly, but prob within
# factor of 2...

# 0.174 s
# 0.25 / 1500 * 0.174 => 29us.
# I think that my calculation of net skew is the time difference
# between a signal propagating in the current vs. in quiescent water.

# hmm.
# so the total distance is something like
# 1500 * 0.174 = 261m
#

# more explicitly:
#  time with a headwind minus the unimpaired time
#  distance      / reduced speed  - unimpaired time
#  (1500 * 0.174) / (1500-0.25)   - 0.174 = 29us
#  So my signal is much smaller -- but the assumption of 0.75m/s and 0.25 m/s is
#  probably an exaggeration, and some of the distance traveled is not parallel to the
#  flow
#  Still need to properly work through the whole formula

if 0:
    # Any chance that filtering out the spikes gives something reasonable?
    lp_skew=net_skew.copy()
    lp_skew[np.abs(lp_skew)>6000]=np.nan

    lp_skew=filters.lowpass_fir(lp_skew,winsize=5000)
    axs[0].plot(t_complete,lp_skew,label='LP Net skew')
axs[0].legend(loc='upper left')

# plot skew and travel in original time coordinate for
# each
def demean(x):
    return x-np.nanmean(x)

for i,leg in enumerate( legs ):
    axs[1].plot(leg.transits.time,leg.transits.clock_skew_us,label=leg.label)
    
    axs[2].plot(leg.transits.time,demean(leg.transits.mean_travel_us),
                label=leg.label)

axs[0].axis(ymin=-100,ymax=100)

axs[1].legend(loc='upper left')

axs[2].axis(ymin=-10000,ymax=10000)
axs[2].legend(loc='upper left')


axs[1].set_ylabel("leg skew (us)")
axs[2].set_ylabel("leg travel anom. (us)")

ax_hist=fig.add_subplot(gs[:,-1])
ax_hist.hist2d( net_skew,net_stale/1e6, bins=200, range=[ [-20,20],[0,1000]],
                cmap='CMRmap_r')
ax_hist.set_xlabel('Net skew us')
ax_hist.set_ylabel('Stale sec')
plt.axvline(0,color='k')

fig.tight_layout()

# fig.savefig('triangle-skew-with_stale.png')

##
plt.figure(20).clf()
fig,axs=plt.subplots(3,1,sharex=True,num=20)

#for i,(leg,skew) in enumerate( zip(legs,skews) ):
#    axs[0].plot(t_complete,skew)

net_skew=skews[0]+skews[1]+skews[2]
axs[0].plot(t_complete,net_skew,label='Net skew~circulation')

# Any chance that filtering out the spikes gives something reasonable?
# what sign would I expect here?

lp_skew=net_skew.copy()
lp_skew[np.abs(lp_skew)>6000]=np.nan

lp_skew=filters.lowpass_fir(lp_skew,winsize=5000)
axs[0].plot(t_complete,lp_skew,label='LP Net skew')


hnd_flow=axs[1].plot(msd_flow.time,msd_flow.sensor0020,label='Flow')

ax_stage=plt.twinx(axs[1])
hnd_stage=ax_stage.plot(msd_stage.time,msd_stage.sensor0001,'g-',label='Stage')

ds=AM3_SM8.B_full
axs[2].plot( ds.time,ds.temp,label='Temperature (AM3)')

axs[0].legend(loc='upper right')
axs[1].legend(handles=hnd_flow+hnd_stage,
              loc='upper right')
axs[2].legend(loc='upper right')

axs[0].axis(ymin=-5000,ymax=5000)

fig.tight_layout()

## 
# fig.savefig('triangle-skew-analysis.png',dpi=150)

##
