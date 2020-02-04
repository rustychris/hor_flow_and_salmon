# Select a candidate track that looks like it would benefit
# from better processing. 
# Pull the respective period of pings
# Fit the clock offsets
# Fit ping-by-ping with triplerx pings
# see where things are.
import seawater
from scipy.special import erfc
from stompy.spatial import linestring_utils

import numpy as np
import xarray as xr
import prepare_pings as pp
from stompy.plot import plot_utils
from stompy import utils
from stompy.spatial import field
from scipy.optimize import fmin, brute
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import collections

##
img=field.GdalGrid("../../model/grid/boundaries/junction-bathy-rgb.tif")
cleaned=pd.read_csv("../tags/cleaned_half meter.csv")

##

dem=field.GdalGrid("../../bathy/junction-composite-dem-no_adcp.tif")

##
import pickle
import pystan

model_file='single_ping.stan'
model_pkl=model_file+'.pkl'

if False: # utils.is_stale(model_pkl,[model_file]):
    sm = pystan.StanModel(file=model_file)
    with open(model_pkl, 'wb') as fp:
        pickle.dump(sm, fp)
else:
    with open(model_pkl,'rb') as fp:
        sm=pickle.load(fp)

## 
def cost2(xy,data):
    rx_xy=np.c_[ data['rx_x'], data['rx_y']]
    dists=utils.dist(xy - rx_xy)
    transits=dists/data['rx_c']
    d_transits=np.diff(transits)
    d_times=np.diff(data['rx_t'])
    
    err=( (d_transits-d_times)**2 ).sum()
    return err

def solve_analytical(data):
    """
    Given a dictionary ready for passing to STAN
    calculate pair of analytical solutions.
    Expects data to have rx_x, rx_y, rx_t, and rx_c.
    
    Note that only the mean value of rx_c is used.

    Based on the soln approach of Mathias et al 2008.
    """
    S=np.eye(3)
    S[-1,-1]=-data['rx_c'].mean()**2

    Nb=len(data['rx_x'])
    
    q_k=[np.array([data['rx_x'][k],
                   data['rx_y'][k],
                   data['rx_t'][k]])
         for k in range(Nb)]

    Qk=np.vstack(q_k)

    QkS=Qk.dot(S)

    arhs=[ q_i.T.dot(S).dot(q_i) for q_i in q_k]

    invQks=np.linalg.pinv(QkS) # or pseudo-inverse

    a=0.5*invQks.dot(arhs)
    b=0.5*invQks.dot(np.ones_like(arhs))

    # at this point q=a+v*b
    # substitute back into definition of v
    # to get coefficients of quadratic equation
    C=a.dot(S).dot(a)
    B=2*a.dot(S).dot(b)-1
    A=b.dot(S).dot(b)
    # Av^2 + B*v + C = 0
    determ=B**2-4*A*C
    if determ<0:
        print("Determinant was negative")
        return []
    
    v1=(-B+np.sqrt(determ))/(2*A)
    v2=(-B-np.sqrt(determ))/(2*A)

    # solutions:
    q1=a+v1*b
    q2=a+v2*b

    rx_t_min=data['rx_t'].min()
    
    # Check times to make sure causality is preserved
    q_valid=[ q for q in [q1,q2] if q[-1]<=rx_t_min]
    # print(f"Analytical solution yielded {len(q_valid)} solutions")

    return q_valid


# Can stan resolve the multiple possible solutions?
    
def solve_tag_position(data):
    #times,c,x,y,z):
    #data=solver_data(times,c,x,y,z)

    # First try to solve it analytically to get good starting points
    solns=solve_analytical(data)

    if len(solns)>0:
        xy_solns=np.array(solns)[:,:-1] # drop times

        # polished=[]
        # for xy_soln in xy_solns:
        #     opt=sm.optimizing(data=data,init=dict(x=xy_soln[0],y=xy_soln[1]))
        #     polished.append( np.array([opt['x'],opt['y']]) )
        # xy_solns=np.array( polished )
    else:
        # if there were no analytical solutions, try just optimizing
        opt=sm.optimizing(data=data)
        xy_solns=np.array([ [opt['x'],opt['y']]])

    # And for testing, just return those.
    xy_solns+=data['xy0']
    return xy_solns

def data_to_hyperbola(data,a=0,b=1):
    s=np.linspace(-5,5,100)

    # Draw hyperbolas from data:
    pa=np.r_[ data['rx_x'][a], data['rx_y'][a] ]
    pb=np.r_[ data['rx_x'][b], data['rx_y'][b] ]

    f0=0.5*(pa+pb)

    dab=pb-pa
    theta=np.arctan2(dab[1],dab[0]) # for the rotation
    focus_dist=utils.dist(dab)

    delta_ab=(data['rx_t'][b] - data['rx_t'][a])*0.5*(data['rx_c'][a]+data['rx_c'][b])

    # well, I can get the eccentricity, right?
    hyp_c=0.5*utils.dist(pa-pb) # center to one focus
    # distance from a point to other focus
    #         distance to this focus
    # (c+a) - (c-a) = delta_ab
    hyp_a=-delta_ab/2

    ecc=hyp_c/hyp_a

    # ecc = sqrt(1+b^2/a^2)
    # ecc^2-1 = b^2/a^2
    # take a^2=1
    # b^2=ecc^2-1
    # b=sqrt( ecc^2-1)
    B=np.sqrt(ecc**2-1)

    hxy_sim=np.c_[np.cosh(s),
                  B*np.sinh(s)]
    if 0:
        hxy_sim_ref=np.c_[-np.cosh(s),
                          B*np.sinh(s)]
        # while working through this, include the reflection
        hxy_sim=np.concatenate( [hxy_sim,
                                 [[np.nan,np.nan]],
                                 hxy_sim_ref])

    # so a point on the hyperbola is at [1,0]
    # and then the focus is at [ecc,0]

    f1=np.array([ecc,0])
    f2=-f1

    deltas=utils.dist( hxy_sim-f1) - utils.dist(hxy_sim-f2)

    # scale it - focus is current at ecc, and I want it at
    # hyp_c
    # hyp_c/ ecc =hyp_a
    hxy_cong=hxy_sim * hyp_a

    hxy_final=utils.rot(theta,hxy_cong) + f0 + data['xy0']

    return hxy_final


##

# Looking through ~/src/hor_flow_and_salmon/analysis/swimming/plots-2019-06-28/plot_cleaned/
fish_tag='7ADB' # decent track with some missing chunks

##

# Read all the receivers to find out when this tag was seen
pm=pp.ping_matcher_2018()

##
t_mins=[]
t_maxs=[]

for rx in pm.all_detects:
    hits=np.nonzero( (rx.tag.values==fish_tag) )[0]
    if len(hits)==0: continue
    t_mins.append( rx.time.values[hits[0]] )
    t_maxs.append( rx.time.values[hits[-1]] )
t_min=np.min(t_mins)
t_max=np.max(t_maxs)

print(f"Time range for tag {fish_tag}: {t_min} -- {t_max}")

# Add some padding before/after to help with getting good clock sync
pad=np.timedelta64(1800,'s')

t_start=t_min-pad
t_stop =t_max+pad
##

pm_nomp=pm.remove_multipath()
pm_clip=pm_nomp.clip_time([t_start,t_stop])

fn=f'pings-{str(pm_clip.clipped[0])}_{str(pm_clip.clipped[1])}.nc'
if not os.path.exists(fn):
    # This takes 2 minutes
    ds_total=pm_clip.match_all()
    ds_total.to_netcdf(fn)

##

# Fit the clock offsets
ds_total=xr.open_dataset(fn)


##

# Can I fit all of them at once?
# For each beacon ping I track two matrices ~ [from,to] delta and
# staleness.
# these get stacked and time deltas are interpolated, while the
# staleness is extrapolated

# Each ping will have a sparse delta matrix, but the interpolation
# fills that in.  From the dense delta matrix, 0.5*(a_ij-a_ji)
# gives the offset
# and 0.5*(a_ij+a_ji) gives the transit time.

# What happens between two receivers who never hear each other?
# Then there would never be an entry in the sparse matrices,
# and thus no values to interpolate.
# Assuming A<->B and B<->C,

# But the offset matrix can be

# Offset matrix (antisymmetric)
#   a   b   c
#a  0  10 [30]
#b      0  20
#c          0

# The question of whether a<->c should be calculated from
# a very stale set of pings or from a transitive shift
# is still ad-hoc.

# Before I was using a reference station, with each station
# taking on a shift relative to that reference.
# That fails in the case of independent groups of receivers,
# though.
# This method would preserve their independence.
# For a single tag, there will be a limited number of
# pings to process, so it's okay if processing each one requires
# some work to get a good set of clock offsets.

# def effective_clock_offset(rxs,ds_tot):
ds=ds_total 

beacon_to_xy=dict( [ (ds.rx_beacon.values[i],
                      (ds.rx_x.values[i],ds.rx_y.values[i]))
                     for i in range(ds.dims['rx'])] )

is_multirx=(np.isfinite(ds.matrix).sum(axis=1)>1).values
is_triplerx=(np.isfinite(ds.matrix).sum(axis=1)>2).values
# it came from one of them.
is_selfaware=np.zeros_like(is_multirx)
tx_beacon=np.zeros(ds.dims['index'],np.int32)

for i,tag in enumerate(ds.tag.values):
    if tag not in beacon_to_xy:
        tx_beacon[i]=-1
    else:
        tx_beacon[i]=np.nonzero(ds.rx_beacon.values==tag)[0][0]

    # And did the rx see itself?
    is_selfaware[i]=np.isfinite(ds.matrix.values[i,tx_beacon[i]])
ds['tx_beacon']=('index',),tx_beacon
is_beacon=tx_beacon>=0

temps=ds.temp.values

# WHOA -- comparisons to the beacon-beacon transits suggest
# that this has some significant error.  The inferred and
# calculated speeds of sound are close-ish if temperature is
# offset 4.5 degC.
rx_c=seawater.svel(0,ds['temp']-4.5,0) 
# occasional missing data...
rx_c=utils.fill_invalid(rx_c,axis=1)

ds['c']=('index','rx'),rx_c

## 
sel_pings=is_multirx&is_selfaware

ds_strong=ds.isel(index=sel_pings)


## 
# Calculate time deltas for each ordered pair of receivers
Nping=ds_strong.dims['index']
Nrx=ds_strong.dims['rx']
tx_beacon=ds_strong.tx_beacon.values

matrix=ds_strong.matrix.values


# delta_mat[i,j,k] is the difference in rx times of ping i
# originating at rx j and received at rx k
delta_mat=np.nan*np.zeros( (Nping,Nrx,Nrx), np.float64)
stale_mat=np.nan*np.zeros( (Nping,Nrx,Nrx), np.float64)

def interp_with_staleness(deltas,stales,tnums):
    missing=np.isnan(deltas)
    if np.all(missing):
        return
    deltas[missing]=np.interp(tnums[missing],
                              tnums[~missing],deltas[~missing])
    stales[missing]=np.abs( tnums[missing] - utils.nearest_val(tnums[~missing],tnums[missing]) )

for a in range(Nrx):
    for b in range(Nrx):
        if a==b: continue # diagonal is not used
        ab_mask=(tx_beacon==a)&(np.isfinite(matrix[:,b]))
        delta_mat[ab_mask,a,b]=matrix[ab_mask,b] - matrix[ab_mask,a]
        stale_mat[ab_mask,a,b]=0.0 # source data not stale

        interp_with_staleness(delta_mat[:,a,b],
                              stale_mat[:,a,b],
                              ds_strong.tnum.values)
## 

mat_tnums=ds_strong.tnum.values
offset_mat=0.5*(delta_mat - delta_mat.transpose(0,2,1)) # antisymmetric
transit_mat=0.5*(delta_mat + delta_mat.transpose(0,2,1)) # symmetric
stale2_mat=stale_mat + stale_mat.transpose(0,2,1)


## And get speed of sounds from here

rx_rx_dists=np.zeros((Nrx,Nrx),np.float64)

for a in range(Nrx):
    xy_a=np.c_[ds.rx_x.values[a], ds.rx_y.values[a]]
    for b in range(a,Nrx):
        xy_b=np.c_[ds.rx_x.values[b], ds.rx_y.values[b]]
        d=utils.dist(xy_a-xy_b)
        rx_rx_dists[a,b]=d
        rx_rx_dists[b,a]=d

measured_c_mat=rx_rx_dists[None,:,:] / transit_mat

##
if 0:
    # This is a bit troubling -- using just temperature,
    # the median speed of sound is 1479 m/s
    # but from the beacon-beacon tags, I get 1463 m/s

    measured_c=measured_c_mat[ np.isfinite(measured_c_mat) ]

    bins=np.linspace(1440,1490,200)

    temp_valid=ds['temp'].values
    temp_valid=temp_valid[ np.isfinite(temp_valid) ]

    plt.figure(2).clf()
    fig,axs=plt.subplots(2,1,num=2,sharex=True)

    # including salinity shifts the distribution
    # faster.  including pressure also makes it
    # faster.
    c_from_temp=seawater.svel(0,temp_valid-4.5,0) 

    axs[0].hist(measured_c,bins=bins)
    axs[1].hist(c_from_temp,bins=bins)

##

# Just pick out the one tag I care about
ds_fish=ds_total.isel(index=is_triplerx & (ds_total.tag.values==fish_tag))

##

matrix=ds_fish.matrix.values
fish_tnums=ds_fish.tnum.values

fixes=[]

def ping_to_solver_data(ping):
    ping_tnum=ping.tnum.item() # fish_tnums[i]
    ping_rx=ping.matrix.values # matrix[i,:]

    offset_idx=utils.nearest(mat_tnums,ping_tnum)
    fish_to_offset=np.abs(ping_tnum-mat_tnums[offset_idx])

    offsets=offset_mat[offset_idx,:,:]
    # total staleness -- include offset to fish time to be conservative.
    stales =stale2_mat[offset_idx,:,:] + fish_to_offset

    # The shortest path traversing offsets based on staleness.
    # this will choose a->b->c over a->c if the collection of
    # pings for the former case are "fresher" in time than
    # for a->c directly.
    dists,preds=shortest_path(stales,return_predecessors=True)
    # dists: [N,N] matrix of total staleness
    # preds: [N,N] of predecessors

    # the specific indices involved in this ping
    rxs=np.nonzero(np.isfinite(ping_rx))[0]

    # and calculate offsets to the other receivers
    # use stales as a distance matrix, and find the
    # shortest path

    # declare rxs[0] as the reference...
    idx0=rxs[0]
    best_offsets=offsets[idx0,:].copy()
    best_offsets[idx0]=0.0 # declare rxs[0] to be the reference time

    for idxn in rxs[1:]:
        stale_sum=0.0
        offset_sum=0.0
        trav=idxn
        while trav!=idx0:
            new_trav=preds[idx0,trav]
            # new_trav-trav is an edge on the shortest path
            stale_sum+=stales[new_trav,trav]
            offset_sum+=offsets[new_trav,trav]
            trav=new_trav

        #print(f"{idx0:2} => {idxn:2}: best stale={stale_sum:.2f}  offset={offset_sum:.4f}")
        #print(f"          orig stale={stales[idx0,idxn]:.2f}  offset={offsets[idx0,idxn]:.4f}")
        best_offsets[idxn]=offset_sum

    ping_rx_adj=(ping_rx - best_offsets) - ping_rx[idx0]

    times=ping_rx_adj[rxs],
    c=ping.c.isel(rx=rxs).values # ds_fish.c.isel(index=i,rx=rxs).values
    x=ping.rx_x.isel(rx=rxs).values
    y=ping.rx_y.isel(rx=rxs).values
    z=ping.rx_z.isel(rx=rxs).values

    time_scale=1000.
    rx_xy=np.c_[x,y]
    xy0=rx_xy.mean(axis=0)
    rx_xy-=xy0
    
    data=dict(Nb=len(x),
              xy0=xy0,
              time_scale=time_scale,
              rx_t=time_scale*ping_rx_adj[rxs],
              rx_x=rx_xy[:,0],
              rx_y=rx_xy[:,1],
              sigma_t=0.1,
              sigma_x=1000.0,
              rx_c=c/time_scale)
    return data

def fish_add_fixes(ds_fish):
    """
    Given the rx data for a specific tag, add fixes for each
    ping where possible
    """
    ds_fish=ds_fish.copy()
    
    for i in range(ds_fish.dims['index']):
        data=ping_to_solver_data(ds_fish.isel(index=i))

        # bundle up the time, speed of sound, and locations for the geometry solution
        tag_locations=solve_tag_position(data)
        for xy in tag_locations:
            fixes.append(dict(x=xy[0],y=xy[1],i=i))
    
    all_fix=pd.DataFrame(fixes)
    ds_fish['fix_idx']=('fix',),all_fix.i
    ds_fish['fix_x']=('fix',),all_fix.x
    ds_fish['fix_y']=('fix',),all_fix.y
    
    return ds_fish

fish2=fish_add_fixes(ds_fish)

## 
def filter_fixes_by_dem(fish,dem,z_max=2.0):
    fish_xy=np.c_[ fish2.fix_x.values, fish2.fix_y.values]
    z=dem(fish_xy)
    invalid=np.isnan(z) | (z>z_max)
    fish=fish.isel(fix=~invalid)
    return fish

fish3=filter_fixes_by_dem(fish2,dem,2.0)

##
max_speed=5.0

def filter_fixes_by_speed(fish,max_speed=5.0):
    # pre-allocate, in case there are indices that
    # have no fixes.
    posns=[ [] ]*fish.dims['index']
    fix_xy=np.c_[ fish.fix_x.values,
                  fish.fix_y.values ]
    for key,grp in utils.enumerate_groups(fish.fix_idx.values):
        posns[key].append(fix_xy[grp])

    for idx in range(fish.dims['index']):
        if len(posns[idx])<2: continue # only filtering out multi-fixes
        
        for pxy in posns[idx]:
            for pxy_previous in posns[idx-1]:
                dist=utils.dist(pxy-pxy_previous)
                dt=HERE - but use YAPS instead

##

this_fish=cleaned[ cleaned.TagID==fish_tag.lower() ]

##

# when there are two solutions, connect by a segment
# This happens 36 times for 7ADB, out of 155 fixes.

segs=[]
fix_xy=np.c_[ fish3.fix_x.values,
              fish3.fix_y.values ]

for key,grp in utils.enumerate_groups(fish3.fix_idx.values):
    if len(grp)==1: continue
    segs.append( fix_xy[grp,:] )

print(f"{len(segs)} locations have multiple solutions")

##

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
scat=ax.scatter(fix_xy[:,0],fix_xy[:,1],12,color='g')
ax.axis('equal')

ax.plot( this_fish.X_UTM, this_fish.Y_UTM, 'k.')

img.plot(ax=ax,zorder=-5)

ax.add_collection(collections.LineCollection(segs,color='g'))
                  
ax.axis('off')
fig.tight_layout()

def pick_idx(idx,**kws):
    print(idx)
    print(kws)
    
plot_utils.enable_picker(scat,ax=ax,cb=pick_idx)
# results at this point are similar but visually not quite as good
# as the original.
# could try pulling c from the clock drift calcs.  no real help.
# could also introduce z, or some allowance for z.

##

# Plot details for a specific location
idx=46 # index into all_fix.

plt.figure(2).clf()
fig,ax=plt.subplots(1,1,num=2)

frame_points=[]

fix_x=all_fix.x[idx]
fix_y=all_fix.y[idx]
fix_i=all_fix.i[idx]
scat=ax.scatter([fix_x],[fix_y],
                12,color='g')
ax.axis('equal')

frame_points.append( [fix_x,fix_y] )

# Show the previous 2 and next 2 locations
bracket_i=range(fix_i-2,fix_i+3)
bracket_fixes=all_fix[ all_fix.i.isin(bracket_i) ]
ax.scatter( bracket_fixes.x, bracket_fixes.y, 30,bracket_fixes.i,
            cmap='seismic')

img.plot(ax=ax,zorder=-5)
                  
ax.axis('off')
fig.tight_layout()

ds_one=ds_fish.isel(index=all_fix.i[idx])
mat=ds_one.matrix.values

rx_segs=[]
rx_xy=np.c_[ ds_one.rx_x.values, ds_one.rx_y.values]

for rx in range(ds_one.dims['rx']):
    if np.isfinite(mat[rx]):
        rx_segs.append( np.array( [ [fix_x,fix_y],
                                    rx_xy[rx]] ) )
        frame_points.append(rx_xy[rx])
ax.add_collection(collections.LineCollection(rx_segs,color='b'))

# Fit a cloud
data=ping_to_solver_data(ds_one)

# There is something about the tdoa approach that tends to push points
# away.  
data['sigma_t']=0.1
data['sigma_x']=5 # This has to be reeled in quite a bit to get a decent fit.

if 0: # fit model in stan sampling
    fit=sm.sampling(data=data,iter=10000,init=lambda: dict(x=fix_x,y=fix_y))
    samples=fit.extract(['x','y'])
    sample_x=samples['x']+data['xy0'][0]
    sample_y=samples['y']+data['xy0'][1]
    ax.plot(sample_x,sample_y,
             'k.',ms=2,alpha=0.1,zorder=5)

if 1: # draw samples for the time error, generate analytical solutions
    soln_samples=[]
    for i in range(50000):
        data_sample=dict(data)
        rx_t=data['rx_t'].copy()
        # Baseline error -- normally distributed noise
        err_dt=np.random.normal(loc=0,scale=3*data['sigma_t'],size=rx_t.shape)
        
        data_sample['rx_t']=rx_t+err_dt
        solns=solve_analytical(data_sample)
        soln_samples+=solns
        if len(soln_samples)>=1000:
            break
    soln_samples=np.array(soln_samples)
    if len(soln_samples):
        soln_samples[:,:2]+= data['xy0']
        ax.plot(soln_samples[:,0],soln_samples[:,1],'k.',ms=2,alpha=0.1,zorder=5)

        xmed=np.median(soln_samples[:,0])
        ymed=np.median(soln_samples[:,1])
        ax.plot([xmed],[ymed],'k.',ms=6,zorder=5)
        
    else:
        print("NO VALID SAMPLES")
        
frame_points=np.array(frame_points)

ax.axis( xmin=frame_points[:,0].min()-10,
         xmax=frame_points[:,0].max()+10,
         ymin=frame_points[:,1].min()-10,
         ymax=frame_points[:,1].max()+10)

for a in range(data['Nb']-1):
    for b in range(a+1,data['Nb']):
        hyp_xy=data_to_hyperbola(data,a,b)
        ax.plot(hyp_xy[:,0],hyp_xy[:,1],'g-')

        
ax.axis( (647058.3994284421, 647621.3325907472, 4185762.540660423, 4186099.105212007) )

##

# So it's fairly straightforward to take sampled
# time errors and generate locations.
# those locations

# idx=46 is suspicious: 4 receivers, and it fails to get a solution.
# maybe some multipath?  I can increase the noise and get some solutions.
# A naive approach to multipath didn't work (including extra error
# 1% of the time).  Something less ad-hoc might, though.

# There are definitely times that I get two solutions and teknologic
# is reporting the worse of the two.  Not super often.


# Possible avenues forward:
#   1. Just use their data, work on behavior
#   2. Add some heuristics to choose solutions
#      a. discount solutions requiring long transit
#      b. discount solutions far from previous/next solutions.
#      c. discount solutions on land.
#   3. Fit sequences of samples
#      a. more robust options for dealing with sample-sample distance.
#      b. options for including estimated ping time, both to correct
#         3-rx solutions and to get estimates for 2-rx solutions


# Potential plan of attach:
#  Try the steps in (2), to get a cloud of potential solutions
#  Those solutions include ping times.
#  Fit a mean and sigma to the time between pings.
#  [could refit the solutions based on ping-ping times?]
#  Fit clouds to 2-ping solutions using that uncertainty in ping times.

