# Extract streamfunction and potential field from hydro
# to define a coordinate system for extrapolation.
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
import xarray as xr

from stompy import utils, filters
from stompy.model.suntans import sun_driver
from stompy.grid import unstructured_grid
from stompy.spatial import wkb2shp, linestring_utils

##

# Use the explicit tracking of cells.  Use original velocities
# for along-stream.  For across, calculate Perot-like velocities
# on the dual grid. This moves vorticity in the original edge velocities
# to divergence on the dual grid with rotated velocities.  Perot
# then filters out the divergence within the dual cell.

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

    ds=g.write_to_xarray()
    ds['u']=('face','xy'),U
    ds['volume']=('face',),V
    ds['Q']=('edge',),Q
    
    return ds


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

# Then a Perot-like calculation on each cell in the dual
def U_perot(g,Q,V):
    cc=g.cells_center()
    ec=g.edges_center()
    normals=g.edges_normals()

    e2c=g.edge_to_cells()
    Uc=np.zeros((g.Ncells(),2),np.float64)
    dist_edge_face=np.nan*np.zeros( (g.Ncells(),g.max_sides), np.float64)

    for c in np.arange(g.Ncells()):
        js=g.cell_to_edges(c)
        for nf,j in enumerate(js):
            # normal points from cell 0 to cell 1
            if e2c[j,0]==c: # normal points away from c
                csgn=1
            else:
                csgn=-1
            dist_edge_face[c,nf]=np.dot( (ec[j]-cc[c]), normals[j] ) * csgn
            # Uc ~ m3/s * m
            Uc[c,:] += Q[j]*normals[j]*dist_edge_face[c,nf]
    Uc /= np.maximum(V,0.01)[:,None]
    return Uc

def rotated_hydro(hydro):
    """
    hydro: xarray Dataset with a grid and edge-centered fluxes in Q.
    returns a new Dataset with the dual grid, 90deg rotated edge velocities,
    and cell-centered vector velocities.
    """
    g=unstructured_grid.UnstructuredGrid.from_ugrid(hydro)
    
    # using centroids yields edges that aren't orthogonal, but the centers
    # are nicely centered in the cells.
    # As long as the grid is nice and orthogonal, it should be okay
    # use true circumcenters.
    gd=g.create_dual(center='circumcenter',create_cells=True,remove_1d=True)

    # Need to get the rotated Q
    en=g.edges_normals() # edge normals that go with Q on the original grid
    enr=utils.rot(np.pi/2,en) 
    dual_en=gd.edges_normals()
    j_orig=gd.edges['dual_edge']
    edge_sign_to_dual=np.round( (enr[j_orig]*dual_en).sum(axis=1) )
    Qdual=hydro.Q.values[j_orig]*edge_sign_to_dual

    Urot_dual=U_perot(gd,Qdual,gd.cells_area())

    ds=gd.write_to_xarray()
    ds['Q']=('edge',),Qdual
    ds['u']=('face','xy'),Urot_dual
    return ds

def steady_streamline_oneway(g,Uc,x0,max_t=3600,max_steps=1000,
                             u_min=1e-3):
    """
    Trace a streamline downstream 
    g: unstructured grid
    Uc: cell centered velocity vectors

    returns Dataset with positions x, cells, times
    """
    # trace some streamlines
    x0=np.asarray(x0)
        
    c=g.select_cells_nearest(x0,inside=True)
    t=0.0 # steady field, start the counter at 0.0
    edge_norm=g.edges_normals()
    edge_ctr=g.edges_center()
    x=x0.copy()
    pnts=[x.copy()]
    cells=[c]
    times=[t]

    e2c=g.edges['cells']
    
    while (t<max_t) and (c>=0):
        dt_max_edge=np.inf # longest time step we're allowed based on hitting an edge
        j_cross=None
        c_cross=None # the cell that would be entered
        j_cross_normal=None

        for j in g.cell_to_edges(c):
            if g.edges['cells'][j,1]==c: # ~checked
                # normals point from cell 0 to cell 1
                csgn=-1
            else:
                csgn=1

            out_normal=csgn*edge_norm[j] # normal of edge j pointing away from cell c

            d_xy_n = edge_ctr[j] - x # vector from xy to a point on the edge

            # perpendicular distance
            dp_xy_n=d_xy_n[0] * out_normal[0] + d_xy_n[1]*out_normal[1]

            if dp_xy_n<0.0: # roundoff error
                dp_xy_n=0.0

            closing=Uc[c,0]*out_normal[0] + Uc[c,1]*out_normal[1]

            # what cell would we be entering?
            if e2c[j,0]==c:
                nbr_c=e2c[j,1]
            elif e2c[j,1]==c:
                nbr_c=e2c[j,0]
            else:
                assert False

            if closing<0: continue # moving away from that edge

            # need to revisit this check
            if len(cells)>1 and nbr_c==cells[-2]:
                # print('Would be reentering cell %d.  Skip that option'%nbr_c)
                continue

            if (dp_xy_n==0.0) and (closing!=0.0):
                print("On edge j=%d, dp_xy_n is zero, and closing is %f"%(j,closing))
            dt_j=dp_xy_n/closing
            if dt_j>0 and dt_j<dt_max_edge:
                j_cross=j
                c_cross=nbr_c
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
        cells.append(c)
        times.append(t)

        if j_cross is not None: # crossing an edge
            c=c_cross
            if c<0:
                break

            # with roundoff, good to make sure that we are properly on the
            # line segment of j_cross
            nodes=g.nodes['x'][ g.edges['nodes'][j_cross] ]
            
            tangent=nodes[1]-nodes[0]
            edgelen=utils.mag(tangent)
            tangent /= edgelen
            
            alpha=np.dot( x-nodes[0], tangent ) / edgelen
            eps=1e-4
            if alpha<eps: 
                print('alpha correction %f => %f'%(alpha,eps))
                alpha=1e-4
            elif alpha>1-eps:
                print('alpha correction %f => %f'%(alpha,1-eps))
                alpha=1-eps
            x=(1-alpha)*nodes[0] + alpha*nodes[1]
            pnts[-1]=x.copy()
            
            umag=utils.mag(Uc[c])
            if umag<=u_min:
                # should only happen with rotate velocities
                # means we hit shore.
                break
        if len(pnts)>=max_steps:
            print("Stopping on max steps")
            break

    ds=xr.Dataset()
    ds['time']=('time',),np.array(times)
    ds['x']=('time','xy'),np.array(pnts)
    ds['cell']=('time',),np.array(cells)
    return ds

def steady_streamline_twoways(g,Uc,x0):
    """
    Trace upstream and downstream with velocities Uc, concatenate
    the results and return dataset.
    """
    ds_fwd=steady_streamline_oneway(g,Uc,x0)
    ds_rev=steady_streamline_oneway(g,-Uc,x0)
    ds_rev.time.values *= -1
    return xr.concat( [ds_rev.isel(time=slice(None,None,-1)), ds_fwd],
                      dim='time' )

##

# get a 2D run
model=sun_driver.SuntansModel.load("../model/suntans/runs/snubby_steady2d_03")

hydro=extract_global(model)

## 
g=unstructured_grid.UnstructuredGrid.from_ugrid(hydro)
g.edge_to_cells()
U=hydro.u.values

# Simply rotating the cell-centered velocities is error prone, as any
# vorticity in the input becomes divergence or convergence in the
# rotated field.

hydro_rot=rotated_hydro(hydro)
g_rot=unstructured_grid.UnstructuredGrid.from_ugrid(hydro_rot)
g_rot.edge_to_cells()
U_rot=hydro_rot['u']

##

# This works fine, with the exception of some convergent edges
# that at least trace, though they don't look great.
alongs=[]
for s in source_ds.sample.values:
    print(s)
    x0=source_ds.x.values[s,:]
    along=steady_streamline_twoways(g,U,x0)
    alongs.append(along)

acrosses=[]
for s in source_ds.sample.values:
    print(s)
    
    x0=source_ds.x.values[s,:]
    across=steady_streamline_twoways(g_rot,U_rot,x0)
    acrosses.append(across)

##

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

g.plot_edges(ax=ax,color='k',lw=0.3)

slc=slice(None,None,50)

ax.add_collection(collections.LineCollection([ds.x.values for ds in alongs[slc]],color='blue',
                                             lw=0.7))

ax.add_collection(collections.LineCollection([ds.x.values for ds in acrosses[slc]],color='green',
                                             lw=0.7))
ax.plot( source_ds.x.values[slc,0],source_ds.x.values[slc,1],'mo')
