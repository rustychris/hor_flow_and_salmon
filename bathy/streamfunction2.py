# Extract streamfunction and potential field from hydro
# to define a coordinate system for extrapolation.

from stompy import utils, filters
from stompy.model.suntans import sun_driver
from stompy.grid import unstructured_grid
from stompy.spatial import wkb2shp, linestring_utils
import matplotlib.pyplot as plt
from matplotlib import collections
import numpy as np
import xarray as xr
import scipy.integrate 

##

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

def steady_streamline_oneway(g,Uc,x0,max_t=3600,max_dist=np.inf):
    # trace some streamlines
    x0=np.asarray(x0)
    t0=0.0
        
    c=g.select_cells_nearest(x0,inside=True)
    t=0.0 # steady field, start the counter at 0.0
    #edge_norm=g.edges_normals()
    #edge_ctr=g.edges_center()
    x=x0.copy()
    pnts=[x.copy()]
    cells=[c] # for debugging track the past cells
    last_cell=c
    last_path=g.cell_path(c)

    # e2c=g.edges['cells']

    def veloc(time,x):
        """ velocity at location x[0],x[1] """
        if last_path.contains_point(x):
            c=last_cell
        else:
            c=g.select_cells_nearest(x,inside=True)
        if c is not None:
            return Uc[c]
        else:
            return np.zeros(2)
        #t=finder(x[0],x[1])
        #u=u_coeffs[t,0]*x[0]+u_coeffs[t,1]*x[1]+u_coeffs[t,2]
        #v=v_coeffs[t,0]*x[0]+v_coeffs[t,1]*x[1]+v_coeffs[t,2]
        #return [u,v]

    ivp=scipy.integrate.RK45(veloc,t0=t0,y0=x0,t_bound=max_t,max_step=1)
    d=0.0
    while ivp.status=='running':
        ivp.step()
        d+=utils.dist( pnts[-1], ivp.y )
        pnts.append(ivp.y.copy())
        if c is not None:
            cells.append(c)
        else:
            cells.append(-1)
            
        if not last_path.contains_point(ivp.y):
            c=g.select_cells_nearest(ivp.y,inside=True)
            if c is not None:
                last_cell=c
                last_path=g.cell_path(c)

        if d>=max_dist:
            break

    pnts=np.array(pnts)
    cells=np.array(cells)
    ds=xr.Dataset()
    ds['x']=('time','xy'),pnts
    ds['cell']=('time',),cells
    return ds

def steady_streamline_twoways(g,Uc,x0):
    ds_fwd=steady_streamline_oneway(g,Uc,x0)
    ds_rev=steady_streamline_oneway(g,-Uc,x0)
    ds_rev.time.values *= -1
    return xr.concate( ds_rev.isel(time=slice(None,None,-1)),
                       ds_fwd, dim='time' )

def steady_streamline_along(x0,hydro):
    Uc=hydro['U']
    g=hydro['g']
    return steady_streamline_twoways(x0,g,Uc)
    
def steady_streamline_across(x0,hydro):
    Uc=hydro['U']
    g=hydro['g']
    Ucrot=utils.rot(np.pi/2,Uc)
    return steady_streamline_twoways(x0,g,Ucrot)

x0=np.array([647111.27074059,4185842.10825328])
# x0=np.array([ 647560.68669735, 4185508.97425484])
#along=steady_streamline_along(x0,hydro)
#across=steady_streamline_across(x0,hydro)
Uc=hydro['U']
g=hydro['g']
ds_fwd=steady_streamline_oneway(g,Uc,x0)
ds_rev=steady_streamline_oneway(g,-Uc,x0)


##
g=hydro['g']

plt.figure(1).clf()
g.plot_edges()
plt.plot(ds_fwd.x.values[:,0],ds_fwd.x.values[:,1],'r-')
plt.plot(ds_rev.x.values[:,0],ds_fwd.x.values[:,1],'g-')
g.plot_cells( mask=np.unique(ds_fwd.cell.values), color='0.8')

#plt.plot(across[:,0],across[:,1],'g-')
plt.axis('equal')

## 
gtri.plot_cells(labeler=lambda i,r: str(i), clip=zoom,facecolor='none',edgecolor='k',
                centroid=True)


# Somehow left is going into 3888, it think it is then going into
# 3887, which it does very briefly, but it jumps from 3887,
# should be going into 3907, but actually goes the other way?
# I think the problem may be that 3888-3887 is a convergent edge
# Uc[3888]
# Out[261]: array([ 0.00319476, -0.00189244])
# 
# In [262]: Uc[3887]
# Out[262]: array([-0.00017572, -0.000887  ])
# 

# Kludge is to stop on small velocities.
# that seems to be working okay.

# so this is a case where the divergence of the rotated field is nonzero.
# at the scale of two adjacent cells, if there is a change in the
# current direction, there isn't a way to represent the potential flow nature of
# the current

# that's not the full story though.  a perturbation of just 1e-9 is enough to
# avoid this issue.
# the cell history is
# [12526, 1690, 1689, 2295, 1689, 2295, 3888, 2295]

# drifts weirdly on the HOR outflow BC.  An artifact of the frictionless run
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

bad=546
x0=source_ds.x.values[546,:]

Uc=tri_hydro['U']
gtri=tri_hydro['g']
Ucrot=utils.rot(np.pi/2,Uc)
#left,l_cells=steady_streamline_oneway(gtri,Ucrot,x0)
right,r_cells=steady_streamline_oneway(gtri,-Ucrot,x0)

#across=steady_streamline_across(x0,tri_hydro)

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

gtri.plot_edges(ax=ax,color='k',lw=0.3)

slc=slice(bad,bad+1)
#ax.add_collection(collections.LineCollection(alongs[slc],color='b'))
#ax.add_collection(collections.LineCollection(acrosses[slc],color='g'))
ax.plot( source_ds.x.values[slc,0],source_ds.x.values[slc,1],'mo')
ax.plot( right[:,0],
         right[:,1],
         'r-o')
gtri.plot_cells(mask=r_cells,centroid=True,labeler=lambda i,r:str(i))

# bad bounce.
# probably a convergent edge.

# For right, -Ucrot, starts in this cell:
#  -Ucrot[5960]
#  array([-0.296335  ,  0.30229126])
# Then here:
#  -Ucrot[5959]
#  array([0.35587373, 0.34833896])

# It's a convergent edge.  Now that I skip the edge that it entered on,
# it is free to go forever.
# So rather than just ignoring the edge it came in on, if that really is
# the winner (may have to be more careful about that calc, too), then
# travel along the edge?

# maybe a postprocessing step on the rotate velocities would help?

# or an analog to Perot that works for tangential velocities?
# 
