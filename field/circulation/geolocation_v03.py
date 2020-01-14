"""
Once pings have been matched, read pings and station locations in and
try to assemble trajectories.

This time, try Stan.

v03: allow sigma_t to be a parameter
"""

import pandas as pd
import xarray as xr
from stompy import utils
import numpy as np
import pystan
import matplotlib.pyplot as plt
from stompy.spatial import field

##

img=field.GdalGrid("../../model/grid/boundaries/junction-bathy-rgb.tif")

## 

ds=xr.open_dataset('pings-2019-04-14T20:00_2019-04-14T22:00.nc')

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
    return xyt0,ping_xyt_template,ds_strong,matrix,rx_xy


# would like to do this:
# sel_pings=is_triplerx | (is_multirx&is_beacon)
# But to keep it a bit more tractable right now, do this:
# sel_pings=is_triplerx
# this is for just getting clock offsets
# sel_pings=is_beacon&is_multirx

##

code = """
data {
 int Np; // number of pings
 int Nb; // number of beacons
 int Nxy_missing; // number of missing ping locations
 int Nshift_missing; // number of missing t_shifts
 int Ntx_t_missing; // number of missing tx times
 
 real rx_t[Np,Nb];
 real rx_x[Nb];
 real rx_y[Nb];
 real max_dist;
 real c;
 
 real tx_x_known[Np-Nxy_missing];
 real tx_y_known[Np-Nxy_missing];
 real tx_t_known[Np-Ntx_t_missing];
 
 real t_shift_known[Nb-Nshift_missing];
 
 int<lower=1,upper=Nb> i_t_missing[Nshift_missing];
 int<lower=1,upper=Nb> i_t_known[Nb-Nshift_missing];
 
 int<lower=1,upper=Np> i_xy_missing[Nxy_missing];
 int<lower=1,upper=Np> i_xy_known[Np-Nxy_missing];
 
 int<lower=1,upper=Np> i_tx_t_missing[Ntx_t_missing];
 int<lower=1,upper=Np> i_tx_t_known[Np-Ntx_t_missing];
}
parameters {
 real t_shift_missing[Nshift_missing];
 
 real tx_x_missing[Nxy_missing];
 real tx_y_missing[Nxy_missing];
 real tx_t_missing[Ntx_t_missing];

 real<lower=0.01,upper=100> sigma_t;
}
transformed parameters {
 real tx_x[Np];
 real tx_y[Np];
 real t_shift[Nb];
 real tx_t[Np];

 tx_x[i_xy_missing]=tx_x_missing;
 tx_y[i_xy_missing]=tx_y_missing;
 tx_x[i_xy_known]=tx_x_known;
 tx_y[i_xy_known]=tx_y_known;
 tx_t[i_tx_t_known]=tx_t_known;
 tx_t[i_tx_t_missing]=tx_t_missing;
 
 t_shift[i_t_missing]=t_shift_missing;
 t_shift[i_t_known]=t_shift_known;
}
model {
  real rx_t_adj;
  real dt;
  real dist;

  // need some limits on this
  sigma_t ~ cauchy(0,5);

  for ( p in 1:Np ) {
    for ( b in 1:Nb ) {
      // but only select this when rx_t[p,b] is not nan
      if ( !is_nan(rx_t[p,b]) ) { 
        // ping p was heard by receiver b
        rx_t_adj=rx_t[p,b] + t_shift[b];

        dt=rx_t_adj-tx_t[p];
        dist = sqrt( square(tx_x[p]-rx_x[b]) + square(tx_y[p]-rx_y[b]) );
        // previously -- and now again
        // tx_t[p] ~ normal(rx_t_adj - dist/c,sigma_t);
        // not sure if there is a material difference, but this doesn't run.
        // dist ~ normal(c*dt,sigma_t);
        // just another shot in the dark
        dt ~ normal(dist/c,sigma_t);
        // maybe this will speed up convergence and rein in crazy
        // tx_x/y.  Encodes an upper bound on ping travel distance
        dist ~ exponential(1./max_dist);
      }
    }
  }
}
"""

# This is surprisingly slow
sm = pystan.StanModel(model_code=code)

##

# Try this in two steps -- first get the time shifts, then solve
# for a single ping location.

# this should get the same answer, but I'm gettin tx_x that appear
# to be scaled up by this much
time_scale=1000 # model time is 1000x actual time

def build_data(ping_xyt_template,matrix,xyt0,rx_xy):
    xy_missing=np.isnan(ping_xyt_template[:,0])
    i_xy_missing=1+np.nonzero(xy_missing)[0]
    i_xy_known=1+np.nonzero(~xy_missing)[0]

    i_beacons=np.arange(matrix.shape[1])
    i_t_known=1+np.nonzero(i_beacons==0)[0]
    i_t_missing=1+np.nonzero(i_beacons!=0)[0]
    t_known=np.zeros(len(i_t_known))

    data=dict(c=1500./time_scale,
              Np=matrix.shape[0],
              Nb=matrix.shape[1],
              Nxy_missing=len(i_xy_missing),
              i_xy_missing=i_xy_missing,
              i_xy_known=i_xy_known,
              Nshift_missing=len(i_t_missing),
              i_t_known=i_t_known,
              i_t_missing=i_t_missing,
              t_shift_known=t_known,
              rx_t=(matrix-xyt0[2])*time_scale,
              rx_x=rx_xy[:,0]-xyt0[0],
              rx_y=rx_xy[:,1]-xyt0[1],
              tx_x_known=ping_xyt_template[~xy_missing,0]-xyt0[0],
              tx_y_known=ping_xyt_template[~xy_missing,1]-xyt0[1],

              Ntx_t_missing=matrix.shape[0], # defaults to all tx times unknown
              i_tx_t_missing=1+np.arange(matrix.shape[0]),
              i_tx_t_known=np.zeros(0,np.int32),
              tx_t_known=np.zeros(0,np.float64),
              
              # sigma_t=0.005*time_scale, # 0.5ms timing precision
              max_dist=400
              )
    return data

##

# Solve for time shifts
xyt0,ping_xyt_template,ds_strong,matrix,rx_xy=model_inputs(is_beacon&is_multirx)
# quite fast
data=build_data(ping_xyt_template,matrix,xyt0,rx_xy)

op_shifts=sm.optimizing(data=data,iter=50000,tol_rel_grad=1e4,
                        tol_rel_obj=0.01)
def set_known_shifts(data):
    data['t_shift_known']=op_shifts['t_shift']
    data['Nshift_missing']=0
    data['i_t_known']=1+np.arange(len(data['t_shift_known']))
    data['i_t_missing']=np.ones(0,np.int32)
##

# Now solve for a single ping.  That was fast and gave
# decent answer.  Try all of the pings.  Also fast, and
# gives a good answer. Hmm - trying that again and it is taking
# a long time to converge. Because I was trying to use a more
# realistic sigma_t.  A sigma_t of 5ms seems crazy large, but
# quickly yields a realistic trajectory.

# Sampling for all at once is not good, though. Rhat
# is terrible (sometimes>1000) and E-BFMI is order 0.001, while
# it *should* be above 0.2.
# 60% of iterations saturated tree depth
sel_unknown=(~is_beacon)&(is_triplerx)
if 0: # limit to a single ping
    idx_unknown=np.nonzero(sel_unknown)[0]
    sel_unknown[idx_unknown[1:]]=False

dummy_xyt0,ping_xyt_template,ds_strong,matrix,rx_xy=model_inputs(sel_unknown)
data=build_data(ping_xyt_template,matrix,xyt0,rx_xy)
set_known_shifts(data)

##

# with sigma_t=5, this is fast and reasonable results.
# can follow up with sigma_t=1, and still fine.
# But trying to go straight to 1.0 and its slow, never
# gets to the same log prob.
data['sigma_t']=5.0
op_rough=sm.optimizing(data=data,iter=200000,tol_rel_grad=1e4,
                       tol_rel_obj=0.001)
data['sigma_t']=1
op_fine=sm.optimizing(data=data,iter=200000,tol_rel_grad=1e4,
                      tol_rel_obj=0.001,init=op_rough)

# that ends with log prob = -41, and 2-4 bad pings

##

# When fitting all pings...
# one chain finishes in 20s.
# other chains are slower...
# 160s for the 2nd to finish
# For a single ping, it's fast (3 seconds?),
# Rhat=1.0.
# The answer isn't very tightly constrained, tho.
# Optimizing found:
#              ('tx_t', array(-3496360.89854282)),
#              ('tx_x', array(21.58989071)),
#              ('tx_y', array(-24.91558453)),
# But sampling found SD for time of 269.71 ms
# SD of x 348.82 and y 216.99.
# And the optimization answer of 21.589 is about the 3%ile
# of the distribution.
# fit=sm.sampling(data=data,iter=10000)

# Maybe the loose results were because sigma_t was 5 ms?
# With this tighter bound, 20% of iterations saturate,
# Rhat is 50...
# Maybe run it longer, and ease up on sigma_t.
# Now 85% saturate. Rhat is 1.6 for x, but 2.2e4 for y.

##

# Now we're getting somewhere.
# With a nice starting point, Rhat is 1.0
# for all 3 parameters, it runs fast, no warnings
# about tree depth.  IQR is 3m for x, 2m for y,
# and a SD of 1.19ms for t.
# What if I constrain sigma_t even more?
# IQR is 0.5m, t SD is 0.23ms.
def init_from_op(chain_id=0):
    d={}
    d['tx_x_missing']=np.atleast_1d(op_fine['tx_x_missing']).copy()
    d['tx_y_missing']=np.atleast_1d(op_fine['tx_x_missing']).copy()
    d['tx_t']=np.atleast_1d(op_fine['tx_t']).copy()

    # Add some jitter between chains.
    scale=1.0
    for k in d:
        d[k] += scale*(np.random.random(d[k].shape)-0.5)
    
    return d


# even starting with pretty good initial points, this is
# pretty slow...  about 10 minutes
# and saturates all iterations
# good news is that Rhat is good for all but one sample.
# maybe didn't need so many iterations.
# what if sigma_t is 5.0, instead of 1.0?
# finishes in <5 minutes, all Rhat are good, and <1%
# saturate. But standard deviations are large. se_mean
# is okay, though.  typ. 3m or so.
data['sigma_t']=5.0
fit=sm.sampling(data=data,iter=10000,
                control=dict(max_treedepth=10),
                init=init_from_op)

## 
plt.figure(1)
samps=fit.extract(inc_warmup=False,pars=['tx_x_missing','tx_y_missing'])
plt.plot(samps['tx_x_missing'][:,0]+xyt0[0],
         samps['tx_y_missing'][:,0]+xyt0[1],
         'r.')

##

# Now look at some 2 ping solutions.
tag='C2BA'
tag_sel=ds_strong.tag.values==tag

# Get the timing from the fine optimization
t_known=op_fine['tx_t'][tag_sel]
x_known=op_fine['tx_x'][tag_sel]
y_known=op_fine['tx_y'][tag_sel]
deltas=np.diff(t_known)

# This is wrong! for pings separated by more than 5s, this isn't
# a fair way to evaluate standard deviation, since there is already
# some averaging going on. this std dev. is therefore biased low.
counts=np.round(deltas/(5*time_scale)).astype(np.int32)
dt_typs=deltas/counts
dt_std=np.std(dt_typs)
dt_mean=np.mean(dt_typs)

# total number of ping intervals spanning the known pings
known_intervals=counts.sum()
n_pad=5
total_intervals=known_intervals+2*n_pad

t_synth=np.nan*np.zeros(1+total_intervals,np.float64)
t_synth_std=np.nan*np.zeros(t_synth.shape)

i_known=n_pad+np.cumsum(np.r_[0,counts])
t_synth[i_known]=t_known
t_synth_std[i_known]=0.0 # "perfect" knowledge

# Fill in the gaps
for i,ip1 in zip(i_known[:-1],i_known[1:]):
    if i+1==ip1: continue # no gap
    t_synth[i+1:ip1] = np.linspace(t_synth[i],t_synth[ip1],1+ip1-i)[1:-1]
    t_synth_std[i+1:ip1] = dt_std # Wrongest
    k=1+ip1-i
    steps=np.arange(k) # say there are 15 intervals.
    steps2=steps*steps[::-1]/k
    t_synth_std[i+1:ip1] = steps2[1:-1]*dt_std # Wrong, but better

dt0=t_synth[n_pad+1]-t_synth[n_pad]
t_synth[:n_pad]=t_synth[n_pad] + np.arange(-n_pad,0)*dt0
t_synth_std[:n_pad]=t_synth_std[n_pad] + (-np.arange(-n_pad,0))*dt_std

dtn=t_synth[-n_pad-1]-t_synth[-n_pad-2]
t_synth[-n_pad:]=t_synth[-n_pad-1] + np.arange(1,n_pad+1)*dtn
t_synth_std[-n_pad:]=t_synth_std[-n_pad-1] + np.arange(1,n_pad+1)*dt_std

x_synth=np.interp(t_synth,
                  t_known,x_known)
y_synth=np.interp(t_synth,
                  t_known,y_known)



if 0:
    i=np.arange(len(t_synth))

    plt.figure(3).clf()
    fig,ax=plt.subplots(1,1,num=3)

    from matplotlib import collections
    segs=np.array( [ [i,t_synth-100*t_synth_std],
                     [i,t_synth+100*t_synth_std] ] ).transpose(2,0,1)
    segs=segs[np.isfinite(segs[:,0,1])]

    lcoll=collections.LineCollection(segs,color='k',lw=5,zorder=-1)
    ax.plot(i,t_synth,'gx')
    ax.add_collection(lcoll)

##

# Find a ping with 2 receivers from that tag.
# 70 more potential pings.
sel_pings=(np.isfinite(ds.matrix).sum(axis=1)==2).values & (ds.tag.values==tag)
# 30 was no good -- ended up with -6.5ms transit time
# By the time I get to just 1 interval away from a known ping, it's 14.25 ms avg.
# transit.  That's fine, but there is clearly some drift that I can't take out
# with this level of analysis.
idxs=np.nonzero(sel_pings)[0]

two_ping_xyts=[]
for idx in idxs:
    sel_pings=np.arange(sel_pings.shape[0])==idx

    # get the data for the model:
    _,ping_xyt_template_double,ds_double,matrix_double,rx_xy_double = model_inputs(sel_pings,xyt0=xyt0)

    # But now we make a guess at the time.  Matrix has the original receive
    # times. shift origin, scale to ms, apply scale, and per-rx shift
    adj_rx=(matrix_double - xyt0[2])*time_scale + op_shifts['t_shift']

    t_bar=np.nanmean(adj_rx) # assumes single ping

    # Where does that fall relative to known pings?

    t_diff=t_bar-t_synth
    best=np.argmin(np.abs(t_bar-t_synth))
    t_ping=t_synth[best]
    t_ping_std=t_synth_std[best]

    n_pings_to_near = np.round( (t_bar-t_ping)/dt_mean )

    if n_pings_to_near!=0:
        print(f"[{idx}] too far out")
        continue

    dt_transit=t_bar-t_ping

    print(f"[{idx}] Transit time of ping: {dt_transit:.3f} ms, std.dev of {t_ping_std:.3f} ms")

    data=build_data(ping_xyt_template_double,matrix_double,xyt0,rx_xy_double)

    # But force the time to be known
    data['Ntx_t_missing']=0
    data['i_tx_t_missing']=np.zeros(0,np.int32)
    data['i_tx_t_known']=np.array([1])
    data['tx_t_known']=np.array([t_ping])
    # Wrong.  sigma_t in the model is independent for each receiver, but here
    # we mean a variance that is on the tx side.
    data['sigma_t']=t_ping_std

    set_known_shifts(data)

    init=dict(tx_x_missing=[x_synth[best]],
              tx_y_missing=[y_synth[best]])
    
    op_two=sm.optimizing(data=data,iter=200000,tol_rel_grad=1e4,
                         tol_rel_obj=0.001,init=init)

    two_ping_xyts.append( [op_two['tx_x'],op_two['tx_y'],t_ping] )
    
##

op=op_fine
def dict_to_params(op):
    shifts=np.zeros(len(op['t_shift'])+1,np.float64)
    shifts[1:]=op['t_shift'] / time_scale
    ping_xyt=ping_xyt_template.copy()
    ping_xyt[:,0]=op['tx_x'] + xyt0[0]
    ping_xyt[:,1]=op['tx_y'] + xyt0[1]
    ping_xyt[:,2]=op['tx_t']/time_scale + xyt0[2]
    return shifts,ping_xyt

#
# When just solving for beacon pings to get shifts...
# seems to get the same shifts.
# and those shifts do result in realistic transit times.
# the shifts are:
#    ('t_shift',
#      array([ -1.9638005 ,  -0.85115702,  -1.62499468,  -1.59300258,
#             -16.89139191,  -3.1747966 ,  -3.82350542,  -6.03432335,
#              -4.08163729,  -2.76865205, -16.83868522])),

# optimizing with the triplerx pings gets to the same time shifts.

# Plot a summary of solution:
shifts,ping_xyt=dict_to_params(op_fine)


fig=plt.figure(1)
fig.clf()
fig.set_size_inches([10,8],forward=True)
fig,axs=plt.subplots(1,1,num=1)
ax=axs
ax.plot( rx_xy[:,0],rx_xy[:,1],'ro', label='Beacon',zorder=-2)
    
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

##

# Show a single combined track with 2 and 3 ping solutions

sel3=(ds_strong.tag.values=='C2BA') & (np.isfinite(ds_strong.matrix).sum(axis=1)>2).values

xyt3=ping_xyt[sel3]

xyt2=np.array(two_ping_xyts)
xyt2[:,2] = xyt2[:,2]/time_scale
xyt2[:,:] = xyt2[:,:]+xyt0[None,:]

xyt_comb=np.concatenate([xyt2,xyt3])

sort=np.argsort(xyt_comb[:,2])
xyt_comb=xyt_comb[sort,:]

# Well, that's pretty rough.
# With two rx solutions with a known tx_t, there is still an ambiguity.
# and I'm seeing lots of samples that flip across that ambiguity.
# It's better when the optimization is seeded with a starting point,
# but there are still lots of ambiguities that go wrong.
# Could go all in and make a chain out of successive solutions, but
# that's getting pretty involved.  Probably better to go to straight
# analytical solutions at that point.  Regardless, should talk to
# Ed, get his thoughts, and probably look at the 2018 data instead
# of 2019.
ax.plot(xyt_comb[:,0],
        xyt_comb[:,1],
        'r-')

##

two_pings=np.c_[ two_ping_utms, 


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

