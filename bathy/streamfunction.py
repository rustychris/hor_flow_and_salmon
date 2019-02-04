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

def cell_divergence(g,c,Q):
    """
    Compute divergence in cell c of grid g based on edge
    fluxes Q, limited to finite entries of Q
    """
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

# get a 2D run
model=sun_driver.SuntansModel.load("../model/suntans/runs/snubby_steady2d_03")

def extract_global(model):
    """
    Join subdomains, extract last timestep of steady output, return 
    dictionary with 
    U: cell center velocities
    V: cell volumes
    Q: edge fluxes
    g: the grid
    """
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
    # normal_sgn=-1
    Q*=-1

    return dict(U=U,V=V,Q=Q,g=g)

hydro=extract_global(model)

##

# Sample datasets
if 1:
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
    source_ds=xr.Dataset()
    source_ds['x']=('sample','xy'),adcp_xy
    source_ds['z']=('sample',),dem(adcp_xy)

##

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

def fix_divergence(g,Q):
    e2c=g.edge_to_cells()

    for j_boundary in np.nonzero(e2c[:,1]<0)[0]:
        # cell_divergence will give the net flow out of
        # c.  positive Q on j_boundary is flow out of
        # c.  
        Q[j_boundary]-=cell_divergence(g, e2c[j_boundary,0],Q)

fix_divergence(hydro['g'],hydro['Q'])


def fill_by_divergence(gtri,Qtri):
    """
    Update nan entries in Qtri based on divergence in adjacent
    cells.
    """
    bad_cells=[]

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

##

def quad_to_triangles(hydro):
    gtri=hydro['g'].copy()

    gtri.make_triangular()

    # Expand the original edge flows to the triangular grid
    Qtri=np.zeros(gtri.Nedges(), np.float64)
    valid=gtri.edges['orig_edge']>=0
    Qtri[valid]=hydro['Q'][gtri.edges['orig_edge'][valid]]
    Qtri[~valid]=np.nan

    # fill in divergence free velocities for the new edges
    fill_by_divergence(gtri,Qtri)

    # Move this earlier as part of initialization
    gtri.add_cell_field('wc_depth', g.cells['wc_depth'][ gtri.cells['orig_cell'] ], on_exists='overwrite')

    Vtri=gtri.cells_area()*gtri.cells['wc_depth']

    return dict(g=gtri,Q=Qtri,V=Vtri)

##

# Qtri has flow for all edges, should be non-divergent,
# and corrected at boundaries.

# Precalculate the mean flow in each triangle via Perot


def U_perot(gtri,Qtri,Vtri):
    cc=gtri.cells_center()
    ec=gtri.edges_center()
    normals=gtri.edges_normals()

    e2c=gtri.edge_to_cells()
    Uc=np.zeros((gtri.Ncells(),2),np.float64)
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
            Uc[c,:] += Qtri[j]*normals[j]*dist_edge_face[c,nf]
    Uc /= np.maximum(Vtri,0.01)[:,None]
    return Uc

tri_hydro=quad_to_triangles(hydro)
tri_hydro['U']=U_perot(tri_hydro['g'],tri_hydro['Q'],tri_hydro['V'])

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
            assert dp_xy_n>=-0.01 #otherwise csgn probably wrong above

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


def steady_streamline_along(x0,hydro):
    fwd=steady_streamline_oneway(gtri,Uc,x0)
    rev=steady_streamline_oneway(gtri,-Uc,x0)
    return np.concatenate( ( rev[::-1],fwd[1:] ) )
def steady_streamline_across(x0,hydro):
    Ucrot=utils.rot(np.pi/2,Uc)
    left=steady_streamline_oneway(gtri,Ucrot,x0)
    right=steady_streamline_oneway(gtri,-Ucrot,x0)
    return np.concatenate( ( left[::-1],right[1:] ) )
    
x0=np.array([647111.27074059,4185842.10825328])
along=steady_streamline_along(x0,tri_hydro)
across=steady_streamline_across(x0,tri_hydro)

## 
plt.figure(1).clf()
gtri.plot_edges()
plt.plot(along[:,0],along[:,1],'r-')
plt.plot(across[:,0],across[:,1],'g-')
plt.axis('equal')


# drifts weirdly on the HOR side.  An artifact of the frictionless run
# and stage BC.
##
alongs=[]
acrosses=[]

for s in source_ds.sample.values:
    print(s)
    
    x0=source_ds.x.values[s,:]
    along=steady_streamline_along(x0,tri_hydro)
    across=steady_streamline_across(x0,tri_hydro)

    alongs.append(along)
    acrosses.append(across)

##

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

gtri.plot_edges(ax=ax)
ax.text(x0[0],x0[1],"x0")

##

# cell 2295 has weird stuff in the across trace.
# suddenly has an along-channel velocity
# is it super shallow?

plt.figure(2).clf()
fig,ax=plt.subplots(1,1,num=2)

gtri.plot_edges(ax=ax)
cntr=gtri.cells_centroid()

ax.quiver( cntr[:,0],cntr[:,1], tri_hydro['U'][:,0], tri_hydro['U'][:,1])
