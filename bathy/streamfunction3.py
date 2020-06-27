# Extract streamfunction and potential field from hydro
# to define a coordinate system for extrapolation.
import os
from shapely import geometry
from stompy.spatial import field

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

from stream_tracer import (steady_streamline_twoways, steady_streamline_oneway,
                           rotated_hydro, StreamDistance)
                           

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
U_rot=hydro_rot['u'].values

##

tracks_fn='sparse-tracks2.pkl'

if not os.path.exists(tracks_fn):
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

    import pickle
    with open(tracks_fn,'wb') as fp:
        pickle.dump( dict(acrosses=acrosses,alongs=alongs,hydro=hydro,hydro_rot=hydro_rot),
                     fp )
else:
    with open(tracks_fn,'rb') as fp:
        d=pickle.load(fp)
        acrosses=d['acrosses']
        alongs=d['alongs']
        hydro=d['hydro']
        hydro_rot=d['hydro_rot']
        
##

if 0:
    plt.figure(1).clf()
    fig,ax=plt.subplots(1,1,num=1)

    g.plot_edges(ax=ax,color='k',lw=0.3)

    slc=slice(None,None,50)

    ax.add_collection(collections.LineCollection([ds.x.values for ds in alongs[slc]],color='blue',
                                                 lw=0.7))

    ax.add_collection(collections.LineCollection([ds.x.values for ds in acrosses[slc]],color='green',
                                                 lw=0.7))
    ax.plot( source_ds.x.values[slc,0],source_ds.x.values[slc,1],'mo')

##



SD=StreamDistance(g=g,U=U,g_rot=g_rot,U_rot=U_rot,alongs=alongs,acrosses=acrosses)
stream_distance=SD.stream_distance

def samples_for_target(x_target,N=500):
    x_along=steady_streamline_twoways(g,U,x_target)
    x_across=steady_streamline_twoways(g_rot,U_rot,x_target)

    # nearby source samples
    dists=utils.mag( x_target-source_ds.x )
    close_samples=np.argsort(dists)[:N]

    close_distances=[]

    for s in close_samples:
        close_distances.append( stream_distance(x_target,s,
                                                x_along=x_along,
                                                x_across=x_across) )
    close_distances=np.array(close_distances)
    ds=xr.Dataset()
    ds['target']=('xy',),x_target.copy()
    ds['target_z']=(), dem(x_target)
    ds['stream_dist']=('sample','st'),close_distances
    ds['sample_z']=('sample',),source_ds.z.values[close_samples]
    ds['sample_xy']=('sample','xy'),source_ds.x.values[close_samples]
    return ds

all_targets=[]
cc=g.cells_center()
for c in range(g.Ncells()):
    print(c)
    x_target=cc[c] + np.array([0.1,0.1])
    target_samples=samples_for_target(x_target)
    all_targets.append(target_samples)
    

import pickle
with open('all-targets2.pkl','wb') as fp:
    pickle.dump(all_targets, fp)


if 0:
    zoom=(647327.9824278749, 647346.4060169846, 4185984.8153818697, 4186000.7397412914)

    plt.figure(1).clf()
    fig,ax=plt.subplots(1,1,num=1)
    #g.plot_edges(ax=ax,color='k',lw=0.3)
    g_rot.plot_edges(ax=ax,color='k',lw=0.3)
    g_rot.plot_edges(ax=ax,color='k',lw=0.3,clip=zoom,labeler=lambda j,r: str(j))
    g_rot.plot_cells(ax=ax,color='0.9',clip=zoom,labeler=lambda i,r: str(i),zorder=-2)
    g_rot.plot_nodes(ax=ax,clip=zoom,labeler=lambda i,r: str(i),zorder=-1)

    ax.plot(ds_rev.x.values[:,0],
            ds_rev.x.values[:,1],
            'g-o')
    ax.plot( [x_target[0]],[x_target[1]],'ro')


    ## 
    plt.figure(1).clf()
    fig,ax=plt.subplots(1,1,num=1)
    g.plot_edges(ax=ax,color='k',lw=0.3)

    ax.plot( source_ds.x.values[slc,0],source_ds.x.values[slc,1],'mo')

    ax.plot( [x_target[0]],[x_target[1]],'ro')
    scat=ax.scatter( source_ds.x.values[close_samples,0],
                     source_ds.x.values[close_samples,1],
                     30,close_distances[:,1] )
    scat.set_clim([-40,40])
    scat.set_cmap('seismic')
    ax.axis( (647301.6747810617, 647489.8763778533, 4185618.735872294, 4185781.4072091985) )
    ##

    # Plot those in the plane:
    plt.figure(2).clf()
    fig,(ax,ax_dem)=plt.subplots(2,1,num=2)

    valid=np.isfinite(close_distances[:,0])

    f=field.XYZField( X=close_distances[valid],
                      F=source_ds.z.values[close_samples][valid] )

    clim=[f.F.min(),f.F.max()]

    scat=ax.scatter(f.X[:,0],f.X[:,1],30,f.F,
                    cmap='jet',clim=clim)

    fg=f.to_grid(nx=500,ny=500,aspect=5)

    fg.plot(zorder=-2,ax=ax,cmap='jet',clim=clim)

    scat2=ax.scatter([0.0],[0.0],50,[dem(x_target)],
                     cmap='jet')
    scat2.set_clim(clim)

    plt.colorbar(scat2)

    # and plot things in geographical coordinates
    sample_xy=source_ds.x.values[close_samples,:]
    sample_axis=[sample_xy[:,0].min(),sample_xy[:,0].max(),
                 sample_xy[:,1].min(),sample_xy[:,1].max()]
    img=dem.crop(sample_axis).plot(ax=ax_dem,alpha=0.7)
    img.set_clim(clim)
    img.set_cmap('jet')

    scat3=ax_dem.scatter( source_ds.x.values[close_samples,0],
                          source_ds.x.values[close_samples,1],
                          30,source_ds.z.values[close_samples] )
    scat3.set_clim(clim)
    scat3.set_cmap('jet')

    ax_dem.plot( [x_target[0]],[x_target[1]],'mo')

    ##

    # some weird points that are on river left that show up as almost
    # mid-channel.

    g.Ncells()
