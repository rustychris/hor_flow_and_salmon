"""
Expand on the code in parse_tek.
and jettison some of the exploratory ground-truthing
code
"""

import pandas as pd
import numpy as np
import datetime
from stompy import filters

import matplotlib.pyplot as plt
from scipy.optimize import fmin

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
    # this had been limited to 160e3, but
    # AM9 has a bad calibration (or maybe it's just really deep)
    df2.loc[ df2.pressure>225e3, 'pressure']=np.nan
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

def remove_multipath(ds):
    # Sometimes a single ping is heard twice (a local bounce)
    # when the same tag is heard in quick succession (<1s) drop the
    # second occurrence.
    # ds: Dataset filtered to a single receiver and a single beacon id.
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
    
    def __init__(self,fnA,fnB):
        self.fnA=fnA
        self.fnB=fnB
        
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
        trim to common valid time period
        """
        valid_start=np.max( [self.A_full.time.values[0],  self.B_full.time.values[0]] )
        valid_end = np.min( [self.A_full.time.values[-1], self.B_full.time.values[-1]] )

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
            scoreAB=np.sqrt( np.mean((self.A_to_B.trav_secs.values)**2) )
            scoreBA=np.sqrt( np.mean((self.B_to_A.trav_secs.values)**2) )
            return scoreAB + scoreBA

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

        # Refine that
        best=fmin(cost,[best_shift,best_drift])
        # be sure the last call uses the best values
        cost(best)

    def select_good_transits(self):
        self.a2b_good=self.A_to_B.isel(index=np.abs(self.A_to_B.trav_secs.values)<5)
        self.b2a_good=self.B_to_A.isel(index=np.abs(self.B_to_A.trav_secs.values)<5)

    def figure_matches(self):
        plt.figure(13).clf()
        plt.plot(self.A_to_B.time,self.A_to_B.trav_secs,alpha=0.1,label='A to B all')
        plt.plot(self.a2b_good.time,self.a2b_good.trav_secs,label='A to B good')
        plt.plot(self.B_to_A.time,self.B_to_A.trav_secs,alpha=0.1,label='B to A all ')
        plt.plot(self.b2a_good.time,self.b2a_good.trav_secs,label='B to A good')
        plt.legend()
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

        a2b_transit_us=np.interp( utils.to_dnum(all_times),
                                  utils.to_dnum(self.a2b_good.time.values),
                                  (self.a2b_good.time.values - self.a2b_good.time_src.values)/np.timedelta64(1,'us'))

        b2a_transit_us=np.interp( utils.to_dnum(all_times),
                                  utils.to_dnum(self.b2a_good.time.values),
                                  (self.b2a_good.time.values - self.b2a_good.time_src.values)/np.timedelta64(1,'us'))

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

    def figure_transit_and_skew(self):
        plt.figure(14).clf()
        fig,axs=plt.subplots(2,1,num=14,sharex=True)

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
SM3="2019 Data/HOR_Flow_TekDL_2019/SM3_187026/20190516/187026_190301_124447_P/sm18-7026190421300.DET"
SM8="2019 Data/HOR_Flow_TekDL_2019/SM8_187028/20190516/187028_190216_175239_P/sm18-7028190421324.DET"
AM4="2019 Data/HOR_Flow_TekDL_2019/AM4_186008/20190517/186008_190517_130946_P/186008_190517_130946_P.DET"

AM7="2019 Data/HOR_Flow_TekDL_2019/AM7_186007/20190517/186007_190217_135133_P/am18-6007190481351.DET"
AM9="2019 Data/HOR_Flow_TekDL_2019/AM9_186004/20190517/186004_190311_144244_P/am18-6004190440924.DET"
SM7="2019 Data/HOR_Flow_TekDL_2019/SM7_187017/20190516/187017_190225_154939_P/sm18-7017190391457.DET"

##
# original test pair
AM1_SM3=BeaconPair(AM1,SM3)

##
# good..
AM3_SM8=BeaconPair(AM3,SM8)
##

# okay - lots of multipath, it seems
SM8_AM4=BeaconPair(SM8,AM4)

##

AM4_AM3=BeaconPair(AM4,AM3)

##

# some issues with one or more of these beacons.
# just much deeper.
# but it looks like maybe it wiggled a lot.
# the travel time has crazy noise, like +-0.3ms
# which is 40cm or so.
AM7_AM9=BeaconPair(AM7,AM9)
# and these never converge on a good clock drift.
# maybe outside the checked range.
AM9_SM7=BeaconPair(AM9,SM7)
SM7_AM7=BeaconPair(SM7,AM7)

##


# What does the combination of all 3 tell us?
# these legs form a CW triangle, and AM4-AM3
# are closer to shore river-right. there should be
# a net CW circulation.  
legs=[AM3_SM8,
      SM8_AM4,
      AM4_AM3]

##

AM1_SM8=BeaconPair(AM1,SM8)
SM8_AM3=BeaconPair(SM8,AM3)
AM3_AM1=BeaconPair(AM3,AM1)

legs=[AM1_SM8,
      SM8_AM3,
      AM3_AM1]
##

t_complete=np.unique( np.concatenate( [ leg.transits.time.values for leg in legs] ) )

skews=[ np.interp( utils.to_dnum(t_complete),
                   utils.to_dnum(leg.transits.time.values), leg.transits.clock_skew_us.values,
                   left=np.nan,right=np.nan)
        for leg in legs]

##

from stompy.io.local import cdec
msd_flow=cdec.cdec_dataset('MSD',start_date=BeaconPair.t0,end_date=BeaconPair.tN,sensor=20,
                           cache_dir='.')
msd_stage=cdec.cdec_dataset('MSD',start_date=BeaconPair.t0,end_date=BeaconPair.tN,sensor=1,
                           cache_dir='.')

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
fig.savefig('triangle-skew-analysis.png',dpi=150)

##
