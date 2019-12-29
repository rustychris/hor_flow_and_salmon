"""
Once pings have been matched, read pings and station locations in and
try to assemble trajectories.

This time, try Stan.
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

# would like to do this:
# sel_pings=is_triplerx | (is_multirx&is_beacon)
# But to keep it a bit more tractable right now, do this:
sel_pings=is_triplerx
# this is for just getting clock offsets
# sel_pings=is_beacon&is_multirx
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

code = """
data {
int Np; // number of pings
int Nb; // number of beacons
int Nxy_missing; // number of missing ping locations

real rx_t[Np,Nb];
real rx_x[Nb];
real rx_y[Nb];
real sigma_t;
real c;

real tx_x_known[Np-Nxy_missing];
real tx_y_known[Np-Nxy_missing];

int<lower=1,upper=Np> i_xy_missing[Nxy_missing];
int<lower=1,upper=Np> i_xy_known[Np-Nxy_missing];

}
parameters {
real t_shift[Nb-1];

real tx_x_missing[Nxy_missing];
real tx_y_missing[Nxy_missing];
real tx_t[Np];

}
transformed parameters {
 real tx_x[Np];
 real tx_y[Np];

 tx_x[i_xy_missing]=tx_x_missing;
 tx_y[i_xy_missing]=tx_y_missing;
 tx_x[i_xy_known]=tx_x_known;
 tx_y[i_xy_known]=tx_y_known;
}
model {
  real rx_t_adj;
  // real dt;
  real dist;

  for ( p in 1:Np ) {
    for ( b in 1:Nb ) {
      // but only select this when rx_t[p,b] is not nan
      if ( !is_nan(rx_t[p,b]) ) { 
        // ping p was heard by receiver b
        if (b>1) {
          // beacon time has a shift
          rx_t_adj=rx_t[p,b] + t_shift[b-1];
        } else {
          rx_t_adj=rx_t[p,b];
        }
        // 
        dist = sqrt( square(tx_x[p]-rx_x[b]) + square(tx_y[p]-rx_y[b]) );
        // previously -- and now again
        tx_t[p] ~ normal(rx_t_adj - dist/c,sigma_t);
        // not sure if there is a material difference, but this doesn't run.
        //dt=rx_t_adj-tx_t[p];
        // dist ~ normal(c*dt,sigma_t);
      }
    }
  }
}
"""

# This is surprisingly slow
sm = pystan.StanModel(model_code=code)
##
xy_missing=np.isnan(ping_xyt_template[:,0])
i_xy_missing=1+np.nonzero(xy_missing)[0]
i_xy_known=1+np.nonzero(~xy_missing)[0]

data=dict(c=1500,
          Np=matrix.shape[0],
          Nb=matrix.shape[1],
          Nxy_missing=len(i_xy_missing),
          i_xy_missing=i_xy_missing,
          i_xy_known=i_xy_known,
          rx_t=matrix-xyt0[2],
          rx_x=rx_xy[:,0]-xyt0[0],
          rx_y=rx_xy[:,1]-xyt0[1],
          tx_x_known=ping_xyt_template[~xy_missing,0]-xyt0[0],
          tx_y_known=ping_xyt_template[~xy_missing,1]-xyt0[1],
          sigma_t=0.0005, # 0.5ms timing precision
          )
op=None
##

#     Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
#     2000  -1.18981e+06   0.000448652       24067.7           1           1     2217   
#     6000       -718801     0.0177103       9023.92           1           1     6545   
#    15000       -184302     0.0218316       20646.8      0.3236      0.3236    16508   
#    50000      -56665.5   0.000741984       861.549      0.4748      0.4748    54910   

# trying BFGS?
#   so far just seems much slower.  doesn't complete an iteration.  try it from the start.
# Newton also slow.  Will keep trying it though.
#
# Could try using ms for time instead of s, so that x,y and t have similar
# scales of variation.
if op is None:
    op=sm.optimizing(data=data,iter=10,tol_rel_grad=1e4,
                     algorithm='Newton')
else:
    op=sm.optimizing(data=data,iter=5000,tol_rel_grad=1e4,init=op,
                     tol_rel_obj=0.001,algorithm='Newton')

def dict_to_params(op):
    shifts=np.zeros(len(op['t_shift'])+1,np.float64)
    shifts[1:]=op['t_shift']
    ping_xyt=ping_xyt_template.copy()
    ping_xyt[:,0]=op['tx_x'] + xyt0[0]
    ping_xyt[:,1]=op['tx_y'] + xyt0[1]
    ping_xyt[:,2]=op['tx_t'] + xyt0[2]
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

# From geolocation_v01.py -- same values.
#    shifts=array([ -1.96425551,  -0.85209644,  -1.62492192,  -1.59288653,
#                  -16.89157486,  -3.17477546,  -3.82348054,  -6.03440981,
#                   -4.08178229,  -2.76880143, -16.83907627])
if 0:
    # running geolocation_v01 with the same subset of pings (triplerx),
    # the initial optimization result is much better, but still not great.
    # going from a cost of 10.15 to 9.08 gets it fairly close to the tekno
    # solution
    shifts,ping_xyt=dict_to_params(op)

    matrix_adj=matrix + shifts[None,:]

    transits=matrix_adj - ping_xyt[:,2,None]

    # just plot non-beacon
    transits[ is_beacon[sel_pings],:]=np.nan
    valid_transits=transits[np.isfinite(transits)]

    # This distribution still looks pretty reasonable.
    plt.figure(2).clf()
    plt.hist(valid_transits,bins=100)

#
# res2=sm.sampling(data=data, iter=2000, chains=1)

# With iter=1000 runs in 830s.
# Report suggests it may not be good, though. All iterations "saturated the maximum
#  depth of the three"
# and
#   WARNING:pystan:Chain 1: E-BFMI = 7.84e-05
#   WARNING:pystan:E-BFMI below 0.2 indicates you may need to reparameterize your model

# I think I'm missing any of the centering code.  So rx_xy is in raw utm.
# And I'm ignoring the known positions of beacon pings.
# Run again with centering.  Still probably missing significant parts of
# the previous setup.
# With the centering, it appears to run way faster, so that's good.
#   WARNING:pystan:Chain 1: E-BFMI = 0.000894
# that's better by an order of magnitude.
# and very few iterations saturate the tree.  that's good
# but the results don't look any more reasonable (i.e. the spread on t_shift is
# smaller than it should be.)

# So I don't yet include the information that I know the tx_xyt of beacon pings
# Oh - and time shifts were not handled at all.

# At the moment I think it is also trying to solve for the location of the
# transmitter pings?
# Not getting very good results

# Something is still off -- the optimization is clearly not good.

##

# Plot a summary of solution:

shifts,ping_xyt=dict_to_params(op)


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
#fig.savefig('c2ba-test-solutions.png')

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

