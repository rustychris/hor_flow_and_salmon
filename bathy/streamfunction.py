# Extract streamfunction and potential field from hydro
# to define a coordinate system for extrapolation.

from stompy import utils, filters
from stompy.model.suntans import sun_driver
from stompy.grid import unstructured_grid
from stompy.spatial import wkb2shp, linestring_utils
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

## 

# get a 2D run
model=sun_driver.SuntansModel.load("../model/suntans/runs/snubby_steady2d_03")

# Get a global grid with cell-centered velocities on it at the end of the
# run.
g=model.grid

U=np.zeros((g.Ncells(),2),np.float64)
V=np.zeros(g.Ncells(),np.float64) # cell volumes
Q=np.zeros(g.Nedges(),np.float64) # edge centered fluxes, m3/s

for proc,map_fn in enumerate(model.map_outputs()):
    avg_fn=os.path.join(model.run_dir, "average.nc.%d"%proc)
    if os.path.exists(avg_fn):
        ds=xr.open_dataset(avg_fn)
    else:
        ds=xr.open_dataset(map_fn)
        
    gsub=unstructured_grid.UnstructuredGrid.from_ugrid(ds)
    
    usub=ds.uc.isel(time=-1,Nk=0).values
    vsub=ds.vc.isel(time=-1,Nk=0).values
    
    U[ds.mnptr.values,0]=usub
    U[ds.mnptr.values,1]=vsub

    if 'U_F' in ds:
        Q[ds.eptr.values] = ds.U_F.isel(time=-1,Nk=0).values
        V[ds.mnptr.values]= ds.Ac.values * (ds.eta.isel(time=-1) + ds.dv.values)
    ds.close()

g.add_cell_field('wc_depth',V/g.cells_area())

# Looks like my edge normals are flipped relative to the file.
# Would like to see why, since I thought I had the same convention
# as suntans.  Could patch unstructured_grid to check for n1/n2, though
# they don't have a standard name so it would be a kludge.
normal_sgn=-1
##
plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
g.plot_cells( values=U[:,0], cmap='seismic',clim=[-0.5,0.5],lw=0.25, edgecolor='k')
ax.axis('equal')

##

import scipy.integrate

def streamline_from_x(g,U,x0,max_t=3600,rotate=False):
    # trace some streamlines
    if rotate:
        U=utils.rot(np.pi/2,U)
        
    def diff_fwd(time,x):
        """ velocity at location x[0],x[1] """
        c=g.select_cells_nearest(x,inside=True)
        if c is None:
            return np.array([0,0])
        return U[c,:]

    def diff_rev(time,x):
        """ reverse velocities """
        return -diff_fwd(time,x)

    pnts=[]
    for diff in [diff_fwd,diff_rev]:
        pnts=pnts[::-1]
        
        ivp=scipy.integrate.RK45(diff,t0=0.,y0=x0,t_bound=max_t,max_step=5)
        while ivp.status=='running':
            ivp.step()
            pnts.append(ivp.y)
            c=g.select_cells_nearest(ivp.y,inside=True)
            if c is None:
                break
            if U[c,0]==U[c,1]==0.0:
                break
    pnts=np.array(pnts)
    return pnts

##
if 0:
    tracks=[]
    # 100 tracks from random cells
    for c in utils.progress(np.random.permutation(g.Ncells())[:100]):
        x0=g.cells_center()[c]
        pnts=streamline_from_x(g,U,x0=x0)
        tracks.append(pnts)

    rot_tracks=[]
    # 100 tracks from random cells
    for c in utils.progress(np.random.permutation(g.Ncells())[:100]):
        x0=g.cells_center()[c]
        pnts=streamline_from_x(g,U,x0=x0,rotate=True)
        rot_tracks.append(pnts)

    # ax.plot(pnts[:,0],pnts[:,1],'m-')
    from matplotlib import collections

    lcoll=collections.LineCollection(tracks,lw=0.5,color='y')
    ax.add_collection(lcoll)

    lrcoll=collections.LineCollection(rot_tracks,lw=0.5,color='m')
    ax.add_collection(lrcoll)

# probably needs to be a simulation without advection of momentum,
# otherwise we can get recirculation zones which do work well
# with a streamfunction.

# How to use these to define distances?
# (a) with the rotated tracks, 

##

# Try some ideas on sparse sample data
if 0:
    import bathy
    dem=bathy.dem()

    # fake, sparser tracks.
    adcp_shp=wkb2shp.shp2geom('sparse_fake_bathy_trackline.shp')
    xys=[]
    for feat in adcp_shp['geom']:
        feat_xy=np.array(feat)
        feat_xy=linestring_utils.resample_linearring(feat_xy,1.0,closed_ring=0)
        feat_xy=filters.lowpass_fir(feat_xy,winsize=6,axis=0)
        xys.append(feat_xy)
    adcp_xy=np.concatenate(xys)
    adcp_xyz=np.c_[ adcp_xy, dem(adcp_xy)]

##

# 2000 points.
# while it would be nice to use the pypart code, the basic
# version gets stuck too easily,  and the ketefian code
# doesn't handle quads.
if 0:
    from stompy.model.pypart import basic
    import six
    six.moves.reload_module(basic)

    class StaticTracer(basic.UgridParticles):
        def __init__(self,grid,Ucell,**kw):
            super(StaticTracer,self).__init__(ncs=[],grid=grid,**kw)
            self.U=Ucell

        def update_velocity(self):
            """
            StaticTracer uses a constant in time velocity field
            """
            self.velocity_valid_time=[0,np.inf]

        def update_particle_velocity_for_new_step(self):
            # A little dicey - this overwrites any memory of convergent edges.
            # so every input interval, it's going to forget
            self.P['u']=self.U[ self.P['c'] ]

    ptm=StaticTracer(g,U)
    ptm.add_particles(x=adcp_xy)
    ptm.set_time(0)

    # Forward - one hour
    times=np.arange(0,3600,5)
    ptm.integrate(times)

    #-- 

    ptm=StaticTracer(g,U,record_dense=True)
    ptm.add_particles(x=[adcp_xy[2028,:]])
    ptm.set_time(0)

    # Forward - one hour
    times=np.arange(0,3600,5)
    ptm.integrate(times)

    #--
    pxy=np.concatenate( [d[0] for d in ptm.dense] )
    plt.plot(pxy[:,0],pxy[:,1],'r-o')
    plt.annotate("start",pxy[0,:])
    plt.annotate("end",pxy[-1,:])

##

# That's got issues.

# quick and dirty.
# if the flow is steady then it should be possible to find a
# unique velocity within each triangular cell, and divergence
# should be zero everywhere.
# in the rotated field is that still true?
# given du/dx+dv/dy=0
#  what can be said about dv/dx-du/dy, which is the curl
#  for that to be zero the flow must be irrotational, and
#  we're probably talking about potential flow at that point.

# freesurface boundaries get zero boundary flux, but we can do
# better.

def cell_divergence(g,c,Q):
    Qcell=0.0
    for jc in g.cell_to_edges(c):
        if np.isnan(Q[jc]): continue

        cells=g.edge_to_cells(jc)
        if cells[0]==c:
            # the edge normal points away from c, we are calculating
            # flow out of the cell, so +1
            sgn=1 
        else:
            sgn=-1
        Qcell+=Q[jc]*sgn
    return Qcell


e2c=g.edge_to_cells()

for j_boundary in np.nonzero(e2c[:,1]<0)[0]:
    # cell_divergence will give the net flow out of
    # c.  positive Q on j_boundary is flow out of
    # c.  
    Q[j_boundary]-=cell_divergence(g, e2c[j_boundary,0],Q)

##

plt.figure(1).clf()
ec=g.edges_center()
normals=g.edges_normals()

plt.quiver(ec[:,0],ec[:,1],Q*normal_sgn*normals[:,0],Q*normal_sgn*normals[:,1])
plt.axis('equal')


##
gtri=g.copy()

gtri.make_triangular()

plt.figure(1).clf()
gtri.plot_edges()
## 
# Expand the original edge flows to the triangular grid
Qtri=np.zeros(gtri.Nedges(), np.float64)
valid=gtri.edges['orig_edge']>=0
Qtri[valid]=Q[gtri.edges['orig_edge'][valid]]
Qtri[~valid]=np.nan

# fill in divergence free velocities for the new edges
bad_cells=[]

def fill_by_divergence(gtri,Qtri):
    e2c=gtri.edge_to_cells()
    valid=np.isfinite(Qtri)
    for j in utils.progress(np.nonzero(~valid)[0]):
        c1,c2=e2c[j]

        # flow out of each cell.
        # positive flow on an edge means out of c1, into c2.
        Qcs=[]
        Qabs=[] # to get a sense of roundoff scale
        
        # csgn relates the net flow out of that cell to what should
        # be put on j.  so a net flow out of c1, Qc>0, means that
        # j should put flow into c1, which is a negative Q on edge
        # j.
        for csgn,c in [(-1,c1),(1,c2)]:
            if c<0:
                continue
            Qc=cell_divergence(gtri,c,Qtri)
            Qcs.append(Qc*csgn)
        if len(Qcs)>1:
            eps=1e-3
            abs_scale=1e-2
            abs_err=np.abs(Qcs[0] - Qcs[1])
            rel_err= abs_err / ( np.mean(np.abs(Qcs)) + eps )
            if (rel_err > eps) and (abs_err>abs_scale):
                print("Flows Q[c=%d]=%f Q[c=%d]=%f don't match"%(c1,Qcs[0],c2,Qcs[1]))
                bad_cells.append(c1)
                bad_cells.append(c2)

        Qtri[j]=np.mean(Qcs)
        assert np.isfinite(Qtri[j])

fill_by_divergence(gtri,Qtri)

##

# Move this earlier as part of initialization
gtri.add_cell_field('wc_depth', g.cells['wc_depth'][ gtri.cells['orig_cell'] ], on_exists='overwrite')

## 
# Qtri has flow for all edges, should be non-divergent,
# and corrected at boundaries.

# Precalculate the mean flow in each triangle via Perot
Uc=np.zeros((gtri.Ncells(),2),np.float64)
normals=gtri.edges_normals()
Vtri=gtri.cells_area()*gtri.cells['wc_depth']
cc=gtri.cells_center()
ec=gtri.edges_center()
e2c=gtri.edge_to_cells()

count_pos=0
count_neg=0
dist_edge_face=np.nan*np.zeros( (gtri.Ncells(),3), np.float64)

for c in np.arange(gtri.Ncells()):
    js=gtri.cell_to_edges(c)
    for nf,j in enumerate(js):
        # normal points from cell 0 to cell 1
        if e2c[j,0]==c: # normal points away from c
            csgn=1
        else:
            csgn=-1
        dist_edge_face[c,nf]=np.dot( (ec[j]-cc[c]), normals[j] ) * csgn
        # Uc ~ m3/s * m
        Uc[c,:] += Qtri[j]*normal_sgn*normals[j]*dist_edge_face[c,nf]
Uc /= np.maximum(Vtri,0.01)[:,None]
##

def steady_streamline_oneway(gtri,Uc,x0,max_t=3600,rotate=False):
    # trace some streamlines
    x0=np.asarray(x0)
    if rotate:
        U=utils.rot(np.pi/2,U)

    c=gtri.select_cells_nearest(x0,inside=True)
    t=0.0 # steady field, start the counter at 0.0
    edge_norm=gtri.edges_normals()
    edge_ctr=gtri.edges_center()
    x=x0.copy()
    pnts=[x.copy()]

    while (t<max_t) and (c>=0):
        dt_max_edge=np.inf # longest time step we're allowed based on hitting an edge
        j_cross=None
        j_cross_normal=None

        for j in gtri.cell_to_edges(c):
            if gtri.edges['cells'][j,1]==c: # ~checked
                # normals point from cell 0 to cell 1
                csgn=-1
            else:
                csgn=1

            out_normal=csgn*edge_norm[j] # normal of edge j pointing away from cell c

            d_xy_n = edge_ctr[j] - x # vector from xy to a point on the edge

            # perpendicular distance
            dp_xy_n=d_xy_n[0] * out_normal[0] + d_xy_n[1]*out_normal[1]
            assert dp_xy_n>=-1e-8 #otherwise csgn probably wrong above

            if dp_xy_n<0.0: # roundoff error
                dp_xy_n=0.0

            closing=Uc[c,0]*out_normal[0] + Uc[c,1]*out_normal[1]

            if closing<0: # moving away from that edge
                continue
            else:
                if (dp_xy_n==0.0) and (closing!=0.0):
                    print("On edge j=%d, dp_xy_n is zero, and closing is %f"%(j,closing))
                dt_j=dp_xy_n/closing
                if dt_j>0 and dt_j<dt_max_edge:
                    j_cross=j
                    dt_max_edge=dt_j

        t_max_edge=t+dt_max_edge
        if t_max_edge>max_t:
            # don't make it to the edge
            dt=max_t-t
            t=max_t
            j_cross=None
        else:
            # this step will take us to the edge j_cross
            dt=dt_max_edge
            t=t_max_edge

        # Take the step
        delta=Uc[c]*dt
        x += delta
        pnts.append(x.copy())

        if j_cross is not None:
            # cross edge j, update time.  careful that j isn't boundary
            # or start sliding on boundary.
            # print "Cross edge"
            if e2c[j_cross,0]==c:
                c=e2c[j_cross,1]
            elif e2c[j_cross,1]==c:
                c=e2c[j_cross,0]
            else:
                assert False
            if Uc[c,0]==Uc[c,1]==0.0:
                # should only happen with rotate velocities
                # means we hit shore.
                break

    pnts=np.array(pnts)
    return pnts

x0=np.array([647111.27074059,4185842.10825328])
fwd=steady_streamline_oneway(gtri,Uc,x0)
rev=steady_streamline_oneway(gtri,-Uc,x0)
pnts=np.concatenate( ( rev[::-1],fwd[1:] ) )
##
Ucrot=utils.rot(np.pi/2,Uc)
left=steady_streamline_oneway(gtri,Ucrot,x0)
right=steady_streamline_oneway(gtri,-Ucrot,x0)

## 
plt.figure(1).clf()
gtri.plot_edges()
plt.plot(pnts[:,0],pnts[:,1],'r-')
plt.plot(right[:,0],right[:,1],'g-')
plt.plot(left[:,0],left[:,1],color='orange')
plt.axis('equal')


# drifts weirdly on the HOR side.  An artifact of the frictionless run
# and stage BC.


