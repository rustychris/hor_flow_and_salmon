# Select a candidate track that looks like it would benefit
# from better processing. 
# Pull the respective period of pings
# Fit the clock offsets
# Fit ping-by-ping with triplerx pings
# see where things are.
import seawater
import numpy as np
import xarray as xr
import prepare_pings as pp
from stompy import utils
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
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

rx_c=seawater.svel(0,ds['temp'],0) 
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

##

# Just pick out the one tag I care about
ds_fish=ds_total.isel(index=is_triplerx & (ds_total.tag.values==fish_tag))

##

matrix=ds_fish.matrix.values
fish_tnums=ds_fish.tnum.values

for i in range(ds_fish.dims['index']):
    ping_tnum=fish_tnums[i]
    ping_rx=matrix[i,:]

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
    break
    # bundle up the time, speed of sound, and locations for the geometry solution
    tag_location=solve_tag_position(times=ping_rx_adj[rxs],
                                    c=ds_fish.c.isel(index=i,rx=rxs).values,
                                    rx_x=ds_fish.rx_x.isel(rx=rxs).values,
                                    rx_y=ds_fish.rx_y.isel(rx=rxs).values,
                                    rx_z=ds_fish.rx_z.isel(rx=rxs).values)

times=ping_rx_adj[rxs]
c=ds_fish.c.isel(index=i,rx=rxs).values
rx_x=ds_fish.rx_x.isel(rx=rxs).values
rx_y=ds_fish.rx_y.isel(rx=rxs).values
rx_z=ds_fish.rx_z.isel(rx=rxs).values
# def solve_tag_position(times,c,x,y,z)

##

rx_xy=np.c_[rx_x,rx_y]
xy0=rx_xy.mean(axis=0)
rx_xy-=xy0
def cost(xyt):
    xy=xyt[:2]
    t=xyt[2]
    
    dists=utils.dist(xy - rx_xy)
    transits=dists/c
    dt=t-times
    err=(dt**2).sum()
    return err

## 
def cost2(xy,data):
    rx_xy=np.c_[ data['rx_x'], data['rx_y']]
    dists=utils.dist(xy - rx_xy)
    transits=dists/data['rx_c']
    d_transits=np.diff(transits)
    d_times=np.diff(data['rx_t'])
    
    err=( (d_transits-d_times)**2 ).sum()
    return err

from scipy.optimize import fmin, brute


##

x0,fval,grid,jout=brute(cost2,[(-200,200),(-200,200)],Ns=200,full_output=1)

# How many local minima are there?
# This doesn't work because the discretization leads to many
# local minima.

##
ctr=jout[1:-1,1:-1]

local_min=( (ctr<jout[0:-2,1:-1])
            & (ctr<jout[2:,1:-1])
            & (ctr<jout[1:-1,0:-2])
            & (ctr<jout[1:-1,2:]) )

##

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)
ax.axis('equal')
ax.pcolormesh( grid[1,0,:],
               grid[0,:,0],
               np.log(jout.T))

ax.plot( rx_xy[:,0],rx_xy[:,1],'go')
ax.plot( [x0[0]],[x0[1]],'ro')

##
# Currently not working...
init=np.r_[ 0, 0, times.min() - np.std(times)]
best_xyt=fmin(cost,init)

# Maybe working.
init2=np.r_[ 0, 0]
best_xy=fmin(cost2,init2)

##

# 

# # How about constructing the hyperbolas?
# p0=rx_xy[0]
# p1=rx_xy[1]
# # how much later 0 saw recieved the ping
# delta01=(times[0]-times[1])*0.5*(c[0]+c[1])
# dist01=utils.dist(p0-p1)
# norm01=utils.to_unit(p1-p0)
# 
# def hyp1(s):
#     if delta01>dist01: # also should be a warning
#         # points fall on a ray from p2
#         return p2+norm01*s
#     elif delta01<-dist01: # should be a warning
#         # points fall on a ray from p1
#         n=utils.to_unit(p1-p0)
#         return p1-norm01*s
#     else:
#         # proper hyperbola
#         p_mid=0.5*(p0+p1)
#         # so if delta01 is positive, 0 saw the ping later,
#         # so move the intercept towards
#         p_int=p_mid*delta01/2*norm01
#     


##


# import sympy
# from sympy.solvers import solve
# from sympy import Symbol,sqrt
# 
# # tag location
# x=Symbol('x')
# y=Symbol('y')
# 
# dists=[ sqrt( (rx_xy[i,0].round()-x)**2 + (rx_xy[i,1].round()-y)**2 )
#         for i in range(rx_xy.shape[0]) ]
# 
# d01=(times[0] - times[1])*0.5*(c[0]+c[1])
# d12=(times[1] - times[2])*0.5*(c[1]+c[2])
# 
# # What if I round those?
# # In this specific case I get no answer, maybe because the ping is slightly too far
# # away. If I round down, I get answers 
# 
# # Round deltas down
# eqs=[dists[0] - dists[1] - (d01-0.5).round(),
#      dists[1] - dists[2] - (d12-0.5).round()]
# 
# # This takes... 1 second with "nice" data.
# # and fails to complete with the above data.
# # is it possible that's because I upgraded sympy?
# result=sympy.solve(eqs,[x,y])
# print(result)
# 
# solns=np.array(result,dtype=np.float64)
# ax.plot(solns[:,0],solns[:,1],'ko')

##

# In this case the solution is poor

# is it sufficient to constrain the solutions to one half-plane
# at a time (per time delta), in order to find all unique solutions?
# not sure.
# nope.  It's possible to have a pair of arms intersect in two locations.

# Can stan resolve the multiple possible solutions?
import pystan
sm = pystan.StanModel(file='single_ping.stan')

##
time_scale=1000.
rx_xy=np.c_[ ds_fish.rx_x.isel(rx=rxs).values,
            ds_fish.rx_y.isel(rx=rxs).values]
xy0=rx_xy.mean(axis=0)
rx_xy-=xy0

data=dict(Nb=3,
          rx_t=time_scale*ping_rx_adj[rxs],
          rx_x=rx_xy[:,0],
          rx_y=rx_xy[:,1],
          sigma_t=1.0,
          rx_c=ds_fish.c.isel(index=i,rx=rxs).values/time_scale)

opt=sm.optimizing(data=data) 

##

# Can I construct a case with two solutions?
rx_xy=np.array( [ [0,10],
                  [10,10],
                  [10,0]], dtype=np.float64)
data=dict(Nb=3,
          rx_x=rx_xy[:,0],
          rx_y=rx_xy[:,1],
          sigma_t=1.0,
          sigma_x=50.0,
          rx_c=1.*np.ones(3))

answer=np.array([12.,-1.])

data['rx_t']=utils.dist(rx_xy-answer)/data['rx_c']
## 
x0,fval,grid,jout=brute(cost2,[(10,25),(-20,5)],Ns=200,full_output=1,
                        args=(data,))
opt=sm.optimizing(data=data) 

##
# This does terribly without sigma_x.
# with sigma_x it at least converges, but not
# very tightly.
# additionally tighterning sigma_t 1.0 ==> 0.1 gives a good distribution.
# sigma_t of 0.01:
#  leads to getting stuck in the farther away distribtion
#  providing the near solution as the initial, it gets stuck there.
data['sigma_t']=0.05
fit=sm.sampling(data=data,iter=10000,init=lambda: dict(x=answer[0],y=answer[1]))
samples=fit.extract(['x','y'])

## 
plt.figure(1).clf()
fig,axs=plt.subplots(1,2,num=1,sharex=True,sharey=True,subplot_kw=dict(adjustable='box'))
ax,ax2=axs
ax.set_aspect(1)
coll=ax.pcolormesh( grid[0,:,0],
                    grid[1,0,:],
                    np.log10(jout.T))
coll.set_clim([-4,-2])
ax.plot( rx_xy[:,0],rx_xy[:,1],'go')
ax.plot( [x0[0]],[x0[1]],'ro')

ax.plot( [opt['x']],[opt['y']],'bo')

# ax.plot(samples['x'],samples['y'],'k.')
ax2.hist2d(samples['x'],samples['y'],bins=[np.linspace(10,25,80),
                                           np.linspace(-20,5,80)])

##

# So the inference works, but its ability to find multiple solutions
# is sensitive to sigma_t. Too small it will get stuck in the solution
# with the largest basin. Too large and the solutions blend together
# Simple constraints on which arm of the hyperbola to use do not
# fix the issue, as a single pair of hyperbola arms can yield two solutions
# (but not more than 2).
# Is there another form of constraint that would isolate the solutions?
# Possibly if I look at the arms of *all* pairs of hyperbola.

# Should be able to do this just with a constrained minimization

gx=np.linspace(-10,15,250)
gy=np.linspace(-5,15,250)
gX,gY=np.meshgrid(gx,gy)
gXY=np.stack([gX,gY],0)


fig_i=10

all_arms=[ [1,1,1],
           [1,1,-1],
           [1,-1,1],
           [1,-1,-1],
           [-1,1,1],
           [-1,1,-1],
           [-1,-1,1],
           [-1,-1,-1]]
for arms in all_arms:
    Arows=[]
    lbs=[]
    ubs=[]
    eqs=[] # for finding an initial point

    from scipy.optimize import LinearConstraint, minimize

    arm_i=0
    for a in range(data['Nb']):
        for b in range(a+1,data['Nb']):
            arm=arms[arm_i]
            arm_i+=1

            # calculate the equation of the line rx[a]-rx[b]
            # dx+ey=1
            mat=np.array( [[ data['rx_x'][a], data['rx_y'][a]],
                           [ data['rx_x'][b], data['rx_y'][b]]] )
            rhs=np.array([1,1])

            d_e=np.linalg.solve(mat,rhs)
            f=1.0
            Arows.append( d_e )
            if arm==1:
                lbs.append(f)
                ubs.append(np.inf)
                eqs.append(f+0.1)
            else:
                lbs.append(-np.inf)
                ubs.append(f)
                eqs.append(f-0.1)

    A=np.array(Arows)
    lbs=np.array(lbs)
    ubs=np.array(ubs)
    eqs=np.array(eqs)

    con=LinearConstraint(A,lbs,ubs)
    # Have to come up with an initial point that satisfies the constraints
    # candidates=rx_xy[:,:] # [N,2]

    candidates=[]
    for sel in [ [0,1],[0,2],[1,2]]:
        x=np.linalg.solve(A[sel,:], eqs[sel])
        candidates.append(x)
    candidates=np.array(candidates)
    mat_c=A.dot(candidates.T) # [constraint, candidate]
    eps=1e-5
    satisfy=np.all( (mat_c>=lbs[:,None]-eps) & (mat_c<=ubs[:,None]+eps), axis=0)
    if not np.any(satisfy):
        print(f"Arms: {arms}  no solution")
        continue
    candidates=candidates[satisfy]
    cand_costs=[cost2(xy,data) for xy in candidates]
    best=np.argmin(cand_costs)
    x0=candidates[best] 

    # meth='SLSQP'
    meth="COBYLA"
    optc=minimize(cost2,x0,method=meth,constraints=con,args=(data,))
    if optc['fun']>1:
        continue

    txt=f"Arms: {arms}\nx: {optc['x']}\nfun: {optc['fun']:6f}"
    print(txt)

    # Plot that.
    mat_c=np.tensordot( A, gXY,1)
    sat_lb=mat_c>=lbs[:,None,None]
    sat_ub=mat_c<=ubs[:,None,None]
    sat=np.all(sat_lb&sat_ub,axis=0)

    plt.figure(fig_i).clf()
    fig,ax=plt.subplots(1,1,num=fig_i)
    fig_i+=1
    ax.pcolormesh(gXY[0,...],
                  gXY[1,...],
                  sat)
    ax.axis('equal')
    ax.plot( rx_xy[:,0],rx_xy[:,1],'go',label='rx')
    ax.plot( [x0[0]],[x0[1]],'ro',label='x0')
    ax.plot( [optc['x'][0]],[optc['x'][1]],'bo',label='opt')
    ax.text(0.05,0.05,txt,transform=ax.transAxes,color='w')
    ax.legend(loc='upper right')
    ctrset=ax.contour( grid[0,:,0],
                       grid[1,0,:],
                       np.log10(jout.T))

# This does come up with [12,-1]
# exactly one no-solution.
# but it fails to find the second solution.
# the best it does is [13.51110626 -3.51110626]
# That's still not it.

# HERE - this should work better...
# plot out the starting points
# It fails because the starting point is closer to the
# wrong minima.

##

def solve_analytical(data):
    # Given a dictionary ready for passing to STAN
    # calculate pair of analytical solutions

    # Based on the soln approach of Mathias et al 2008.
    S=np.eye(3)
    S[-1,-1]=-data['rx_c'].mean()**2

    q_k=[np.array([data['rx_x'][k],
                   data['rx_y'][k],
                   data['rx_t'][k]])
         for k in range(3)]

    Qk=np.vstack(q_k)

    QkS=Qk.dot(S)

    arhs=[ q_i.T.dot(S).dot(q_i) for q_i in q_k]

    invQks=np.linalg.inv(QkS) # or pseudo-inverse

    a=0.5*invQks.dot(arhs)
    b=0.5*invQks.dot(np.ones_like(arhs))

    # at this point q=a+v*b
    # substitute back into definition of v
    # to get coefficients of quadratic equation
    C=a.dot(S).dot(a)
    B=2*a.dot(S).dot(b)-1
    A=b.dot(S).dot(b)
    # Av^2 + B*v + C = 0
    v1=(-B+np.sqrt(B**2-4*A*C))/(2*A)
    v2=(-B-np.sqrt(B**2-4*A*C))/(2*A)

    # solutions:
    q1=a+v1*b
    q2=a+v2*b

    rx_t_min=data['rx_t'].min()
    
    # Check times to make sure causality is preserved
    q_valid=[ q for q in [q1,q2] if q[-1]<=rx_t_min]

    return q_valid
    

## 
plt.figure(1).clf()
fig,axs=plt.subplots(1,2,num=1,sharex=True,sharey=True,subplot_kw=dict(adjustable='box'))
ax,ax2=axs
ax.set_aspect(1)
coll=ax.pcolormesh( grid[0,:,0],
                    grid[1,0,:],
                    np.log10(jout.T))
coll.set_clim([-4,-2])
ax.plot( rx_xy[:,0],rx_xy[:,1],'go')
ax.plot( [x0[0]],[x0[1]],'ro')

ax.plot( [opt['x']],[opt['y']],'bo')

# ax.plot(samples['x'],samples['y'],'k.')
ax2.hist2d(samples['x'],samples['y'],bins=[np.linspace(10,25,80),
                                           np.linspace(-20,5,80)])

qi=np.vstack( [q1,q2] )
ax.plot(qi[:,0],
        qi[:,1],
        'ro')

# BUENO!
