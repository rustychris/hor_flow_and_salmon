# Local Variables:
# python-shell-interpreter: "/opt/anaconda3/bin/ipython"
# python-shell-interpreter-args: "--simple-prompt --matplotlib=agg"
# End:
"""
Extract point velocity estimates.
This version adapted to yaps output
"""
import glob, os
import matplotlib
import shutil
import glob
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

from stompy.model.suntans import sun_driver
from stompy.plot import plot_utils
from stompy import utils,memoize
from stompy.grid import unstructured_grid
from stompy.grid import ugrid

import pandas as pd
import xarray as xr
import numpy as np

import track_common
##
import six
six.moves.reload_module(track_common)

if 1: # for fish tracks
    df_in=track_common.read_from_folder('merged_v00')
    output_path='mergedhydro_v00_sun'
else: # for ADCP data
    df_in=track_common.read_from_folder('adcp_2018')
    output_path='adcp_2018_hydro'

col_in='track'
col_out='track'

use_ptm_output=False

if use_ptm_output:
    avg_fns=glob.glob("E:/Home/rustyh/SanJoaquin/model/suntans/cfg008/*/*.nc")
    avg_fns.sort()
    avg_ncs=[]
    for fn in avg_fns:
        nc=xr.open_dataset(fn)
        nc['eta']=nc['Mesh2_sea_surface_elevation']
        nc['Mesh2_face_depth'].attrs['standard_name']="sea_floor_depth"
        nc['Mesh2_face_depth'].attrs['location']='face' # probably not standard
        nc['dv'] = nc['Mesh2_face_depth'] # pos-down
        nc.rename({'nMesh2_data_time':'time'},inplace=True)
        #rename_dims({'nMesh2_data_time':'time'},
        #               inplace=True)
        nc['time']=nc['Mesh2_data_time']
        nc['Mesh2_layer_3d'].attrs['standard_name']='ocean_zlevel_coordinate'
        nc['Mesh2_layer_3d'].attrs['positive']='down' # why was this up at first?
        # dirty, but this will actually add the necessary ugrid fields
        # to the netcdf dataset
        g=unstructured_grid.UnstructuredGrid.read_ugrid(nc,dialect='fishptm')
        cc=g.cells_center()
        nc['xv']=('nMesh2_face',), cc[:,0]
        nc['yv']=('nMesh2_face',), cc[:,1]
        
        # And I need to fabricate uc,vc - but do that in the snap_for_time
        # code.
        avg_ncs.append(nc)
    run_starts=[nc['time'].values[0] for nc in avg_ncs]    
else:                          
    # mod=sun_driver.SuntansModel.load('/opt2/san_joaquin/cfg008/cfg008_20180409')
    # New run, attached via USB to laptop:
    mod=sun_driver.SuntansModel.load('/media/rusty/80c8a8ec-71d2-4687-aa6b-41c23f557be8/san_joaquin/cfg010/cfg010_20180409')
    # Or on cws-linuxmodeling:
    # mod=sun_driver.SuntansModel.load('/opt2/san_joaquin/cfg010/cfg010_20180409')
    
    seq=mod.chain_restarts()
    run_starts=np.array([mod.run_start for mod in seq])

#%%


@memoize.memoize(lru=10)
def map_for_run(run_i):
    if use_ptm_output:
        nc=avg_ncs[run_i]
        return nc
    else:
        outputs=seq[run_i].map_outputs()
        assert len(outputs)==1,"Expecting merged output"
        return xr.open_dataset(outputs[0])

def add_U_perot(snap):
    cc=g.cells_center()
    ec=g.edges_center()
    normals=g.edges_normals()
    
    e2c=g.edge_to_cells()
    
    Uclayer=np.zeros( (g.Ncells(),snap.dims['nMesh2_layer_3d'],2), np.float64)
    Qlayer=snap.h_flow_avg.values
    Vlayer=snap.Mesh2_face_water_volume.values
    
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
            Uclayer[c,:,:] += Qlayer[j,:,None]*normals[j,None,:]*dist_edge_face[c,nf]
            
    Uclayer /= np.maximum(Vlayer,0.01)[:,:,None]
    snap['uc']=('nMesh2_layer_3d','nMesh2_face'),Uclayer[:,:,0].transpose()
    snap['vc']=('nMesh2_layer_3d','nMesh2_face'),Uclayer[:,:,1].transpose()

@memoize.memoize(lru=50)
def snap_for_time(run_i,t_i):
    map_ds=map_for_run(run_i)
    snap=map_ds.isel(time=t_i)
    if use_ptm_output:
        add_U_perot(snap)
    return snap

@memoize.memoize(lru=10)
def ug_for_run(run_i):
    # have to give it a bit of help to figure out the eta variable.
    return ugrid.UgridXr(map_for_run(run_i),
                         face_eta_vname='eta')

@memoize.memoize()
def get_z_bed():
    return -map_for_run(0)['dv'].values

@memoize.memoize()
def smoother():
    ug=ug_for_run(run_i=0)
    return ug.grid.smooth_matrix(f=0.5)


@memoize.memoize(lru=60)
def z_average_weights(run_i,t_i,ztop=None,zbottom=None,dz=None):
    ug=ug_for_run(run_i)
    return ug.vertical_averaging_weights(time_slice=t_i,ztop=ztop,zbottom=zbottom,dz=dz).T

eps=2.0 # finite difference length scale

from scipy.spatial import Delaunay

@memoize.memoize()
def pnts_for_interpolator():
    # cell center points
    ds_map=map_for_run(0)
    pnts=np.c_[ds_map.xv.values,ds_map.yv.values] # could be moved out of here.
    dt=Delaunay(pnts)
    return dt

def quantize_time(t):
    run_i=np.searchsorted(run_starts,t)-1
    ug=ug_for_run(run_i)
    if use_ptm_output:
        time_idx=np.searchsorted(ug.nc.time.values,t)
    else:
        time_idx=utils.nearest(ug.nc.time.values, t)
    return run_i,time_idx

@memoize.memoize(lru=60)
def interpolator(run_i,t_i,ztop=None,zbottom=None,dz=None):
    """
    Returns a function that takes [...,2] coordinate arrays on 
    input, and for the nearest time step of output spatially
    interpolates uc,vc,z_eta,z_bed.
    """
    ug=ug_for_run(run_i)
    z_bed=get_z_bed()

    snap=snap_for_time(run_i,t_i)

    pnts=pnts_for_interpolator()
    
    # Initialize the fields we'll be filling in
    z_eta=snap['eta'].values

    # depth average, or subset thereof
    weights=z_average_weights(run_i,t_i,ztop=ztop,zbottom=zbottom,dz=dz)
    u3d=snap.uc.values
    v3d=snap.vc.values
    u2d=np.where(np.isnan(u3d),0,weights*u3d).sum(axis=0)
    v2d=np.where(np.isnan(v3d),0,weights*v3d).sum(axis=0)
    values=np.c_[u2d,v2d,z_eta,z_bed]
    return LinearNDInterpolator(pnts, values)

##

if not os.path.exists(output_path):
    os.makedirs(output_path)

results=[]

z_slices=[('_top1m',dict(ztop=0,dz=1.0)),
          ('_top2m',dict(ztop=0,dz=2.0)),
          ('_davg',dict(ztop=0,zbottom=0))]

for idx,row in df_in.iterrows(): #track_fn in track_fns:
    #output_fn=os.path.join(output_path,os.path.basename(track_fn))
    print(idx)
    
    track=row[col_in]
    
    if len(track)<2:
        results.append(None)
        continue 

    track=track.copy()
    for fld in ['x','y','tnum']:
        fld_m=0.5*( track[fld].values[:-1] +
                   track[fld].values[1:] )
        track[fld+'_m']=np.r_[ fld_m, np.nan ]
        
    track['time_m']=utils.unix_to_dt64(track['tnum_m'].values)
    
    new_records=[]
    
    for i,seg in utils.progress( track.iloc[:-1,:].iterrows() ):
        rec={} # new fields to be added to segments.
        t=seg['time_m'].to_datetime64()
        run_i,t_i=quantize_time(t)

        rec['index']=i

        for slice_name,slice_def in z_slices:
            Uint=interpolator(run_i,t_i,**slice_def)
        
            # vorticity at centers
            x_samp=np.array( [ [seg['x_m']    , seg['y_m']      ],
                               [seg['x_m']-eps, seg['y_m']      ],
                               [seg['x_m']+eps, seg['y_m']      ],
                               [seg['x_m']    , seg['y_m'] - eps],
                               [seg['x_m']    , seg['y_m'] + eps]
            ] )

            u=Uint(x_samp)
            u=np.where(np.isfinite(u),u,0)
            # central differencing
            #  dv/dx - du/dy
            w=(u[2,1]-u[1,1])/(2*eps) - (u[4,0]-u[3,0])/(2*eps)

            rec['model_vor'  +slice_name]=w
            rec['model_u'    +slice_name]=u[0,0]
            rec['model_v'    +slice_name]=u[0,1]
        
        rec['model_z_eta']=u[0,2]
        rec['model_z_bed']=u[0,3]

        new_records.append(rec)

    # add a nan row since we pad out to the number of positions
    rec={}
    for k in new_records[-1]:
        rec[k]=None
    new_records.append(rec)
    
    model_data=pd.DataFrame(new_records).set_index('index')
        
    for col in model_data.columns:
        track[col]=model_data[col]

    results.append(track) #track.to_csv(output_fn,index=False)

df_in[col_out]=results

track_common.dump_to_folder(df_in,output_path)

## 
# ##
# if 1: # quiver with all stations
#     fig=plt.figure(2)
#     fig.clf()
#     ax=fig.add_axes([0,0,1,1])
#     mod.grid.plot_edges(lw=0.5,ax=ax,color='0.75',zorder=-3)
#     plt.setp(ax.get_xticklabels(),visible=0)
#     plt.setp(ax.get_yticklabels(),visible=0)
# 
#     color_by='|u|'
#     # color_by='Time (d)'
#     
#     if color_by=='|u|':
#         scal=utils.mag(np.c_[joined.model_u.values,joined.model_v.values])
#     else:
#         scal=(joined.time.values-joined.time.values.min())/np.timedelta64(1,'D')
#     quiv=ax.quiver(joined.xm.values,
#                    joined.ym.values,
#                    joined.model_u.values,
#                    joined.model_v.values,
#                    scal,
#                    cmap='jet',
#                    scale=30.0)
#     plt.colorbar(quiv,label=color_by)
#     ax.quiverkey(quiv,0.1,0.1,0.5,"0.5 m/s - depth avg")
#     ax.axis( (647091.6894404562, 647486.7958566407, 4185689.605777047, 4185968.1328291544) )
#     fig.savefig('model-extracted-velocities.png',dpi=200)
# 
# if 1: # scatter with vorticity
#     fig=plt.figure(2)
#     fig.clf()
#     ax=fig.add_axes([0,0,1,1])
#     mod.grid.plot_edges(lw=0.5,ax=ax,color='0.75',zorder=-3)
#     plt.setp(ax.get_xticklabels(),visible=0)
#     plt.setp(ax.get_yticklabels(),visible=0)
# 
#     scat=ax.scatter(joined.xm.values,
#                     joined.ym.values,
#                     15,joined['model_vor'],
#                     cmap='PuOr')
#     scat.set_clim([-0.05,0.05])
#     plt.colorbar(scat,label="Vorticity, depth avg (s$^{-1}$)")
#     ax.axis( (647091.6894404562, 647486.7958566407, 4185689.605777047, 4185968.1328291544) )
#     fig.savefig('model-extracted-vorticity.png',dpi=200)
# 
# # surface plots
# 
# if 1: # quiver with all stations
#     fig=plt.figure(2)
#     fig.clf()
#     ax=fig.add_axes([0,0,1,1])
#     mod.grid.plot_edges(lw=0.5,ax=ax,color='0.75',zorder=-3)
#     plt.setp(ax.get_xticklabels(),visible=0)
#     plt.setp(ax.get_yticklabels(),visible=0)
# 
#     color_by='|u|'
#     # color_by='Time (d)'
#     if color_by=='|u|':
#         scal=utils.mag(np.c_[joined.model_u_surf.values,
#                              joined.model_v_surf.values])
#     else:
#         scal=(joined.time.values-joined.time.values.min())/np.timedelta64(1,'D')
# 
#     quiv=ax.quiver(joined.xm.values,
#                    joined.ym.values,
#                    joined.model_u_surf.values,
#                    joined.model_v_surf.values,
#                    scal,
#                    cmap='jet',
#                    scale=30.0)
#     plt.colorbar(quiv,label=color_by)
#     ax.quiverkey(quiv,0.3,0.1,0.5,"0.5 m/s - surface")
#     ax.axis( (647091.6894404562, 647486.7958566407, 4185689.605777047, 4185968.1328291544) )
#     fig.savefig('model-extracted-surf_velocities.png',dpi=200)
# 
# if 1: # scatter with vorticity
#     fig=plt.figure(2)
#     fig.clf()
#     ax=fig.add_axes([0,0,1,1])
#     mod.grid.plot_edges(lw=0.5,ax=ax,color='0.75',zorder=-3)
#     plt.setp(ax.get_xticklabels(),visible=0)
#     plt.setp(ax.get_yticklabels(),visible=0)
# 
#     scat=ax.scatter(joined.xm.values,
#                     joined.ym.values,
#                     15,joined['model_vor_surf'],
#                     cmap='PuOr')
#     scat.set_clim([-0.05,0.05])
#     plt.colorbar(scat,label="Vorticity, surface (s$^{-1}$)")
#     ax.axis( (647091.6894404562, 647486.7958566407, 4185689.605777047, 4185968.1328291544) )
#     fig.savefig('model-extracted-surf_vorticity.png',dpi=200)
    
##
