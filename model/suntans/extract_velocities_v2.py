# Local Variables:
# python-shell-interpreter: "/opt/anaconda3/bin/ipython"
# python-shell-interpreter-args: "--simple-prompt --matplotlib=agg"
# End:
"""
Extract point velocity estimates.
"""
import matplotlib
matplotlib.use('Agg')
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


##

# extract a velocity for each of the detections.
# segment dnums appear to be in PDT.
# with some quantization of about 30 seconds.
# And what timezone is the model in? UTC.

# 2020-01-21: change from segments_m.csv to segments.csv
# per updated file from Ed.
inp_fn="../../analysis/swimming/code/segments.csv"
seg_dnum_to_utc=7./24 # add this to segment dnums to get utc.

segments=pd.read_csv(inp_fn)

# choose the last run of a sequence:
#mod=sun_driver.SuntansModel.load('runs/snubby_cfg003_20180411')
# this run had a bad BC
#mod=sun_driver.SuntansModel.load('runs/cfg006_20180310_20180310')
# trying again...
#mod=sun_driver.SuntansModel.load('runs/cfg007_20180409')
# This time with proper friction
mod=sun_driver.SuntansModel.load('/opt2/san_joaquin/cfg008/cfg008_20180409')

seq=mod.chain_restarts()
run_starts=np.array([mod.run_start for mod in seq])

all_stations_ds=[]

segments['time']=utils.to_dt64(segments.dnm.values+seg_dnum_to_utc)

##


@memoize.memoize(lru=10)
def map_for_run(run_i):
    outputs=seq[run_i].map_outputs()
    assert len(outputs)==1,"Expecting merged output"
    return xr.open_dataset(outputs[0])

@memoize.memoize(lru=50)
def snap_for_time(run_i,t):
    map_ds=map_for_run(run_i)
    time_idx=utils.nearest(map_ds.time.values, t)
    return map_ds.isel(time=time_idx)

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

@memoize.memoize(lru=20)
def depth_avg(run_i,t):
    ug=ug_for_run(run_i)
    time_idx=utils.nearest(ug.nc.time.values, t)
    # have to transpose as the velocities have Nk first, but
    # this defaults to Nk second
    return ug.vertical_averaging_weights(time_slice=time_idx,ztop=0,zbottom=0).T

@memoize.memoize(lru=20)
def surf_50cm(run_i,t):
    ug=ug_for_run(run_i)
    time_idx=utils.nearest(ug.nc.time.values, t)
    # have to transpose as the velocities have Nk first, but
    # this defaults to Nk second
    return ug.vertical_averaging_weights(time_slice=time_idx,ztop=0,dz=0.5).T

#eps=0.5 # finite difference length scale
eps=2.0

from scipy.spatial import Delaunay

@memoize.memoize()
def pnts_for_interpolator():
    # cell center points
    ds_map=map_for_run(0)
    pnts=np.c_[ds_map.xv.values,ds_map.yv.values] # could be moved out of here.
    dt=Delaunay(pnts)
    return dt

@memoize.memoize(lru=20)
def interpolator(run_i,t):
    """
    Returns a function that takes [...,2] coordinate arrays on 
    input, and for the nearest time step of output spatially
    interpolates uc,vc,z_eta,z_bed.
    """
    ug=ug_for_run(run_i)

    z_bed=get_z_bed()

    snap=snap_for_time(run_i,t)

    pnts=pnts_for_interpolator()
    
    # Initialize the fields we'll be filling in
    z_eta=snap['eta'].values

    # depth average
    weights=depth_avg(run_i,t)
    u3d=snap.uc.values
    v3d=snap.vc.values
    u2d=np.where(np.isnan(u3d),0,weights*u3d).sum(axis=0)
    v2d=np.where(np.isnan(v3d),0,weights*v3d).sum(axis=0)
    values=np.c_[u2d,v2d,z_eta,z_bed]
    return LinearNDInterpolator(pnts, values)

@memoize.memoize(lru=20)
def surf_interpolator(run_i,t):
    """
    Returns a function that takes [...,2] coordinate arrays on 
    input, and for the nearest time step of output spatially
    interpolates uc,vc integrated over the top 50cm of the
    watercolumn
    """
    ug=ug_for_run(run_i)
    snap=snap_for_time(run_i,t)
    pnts=pnts_for_interpolator()
    
    # top 50cm
    weights=surf_50cm(run_i,t)
    u3d=snap.uc.values
    v3d=snap.vc.values
    u2d=np.where(np.isnan(u3d),0,weights*u3d).sum(axis=0)
    v2d=np.where(np.isnan(v3d),0,weights*v3d).sum(axis=0)
    values=np.c_[u2d,v2d]
    return LinearNDInterpolator(pnts, values)

##

def get_new_records():
    new_records=[]

    for i,seg in segments.iterrows():
        print(i)
        rec={} # new fields to be added to segments.
        t=seg['time'].to_datetime64()
        run_i=np.searchsorted(run_starts,t)-1
        snap=snap_for_time(run_i,t)

        Uint=interpolator(run_i,t)
        
        # vorticity at centers
        x_samp=np.array( [ [seg['xm']    , seg['ym']      ],
                           [seg['xm']-eps, seg['ym']      ],
                           [seg['xm']+eps, seg['ym']      ],
                           [seg['xm']    , seg['ym'] - eps],
                           [seg['xm']    , seg['ym'] + eps]
        ] )

        u=Uint(x_samp)
        u=np.where(np.isfinite(u),u,0)
        # central differencing
        w=(u[2,1]-u[1,1])/(2*eps) - (u[4,0]-u[3,0])/(2*eps)

        rec['index']=i
        rec['model_vor']=w
        rec['model_u']=u[0,0]
        rec['model_v']=u[0,1]
        rec['model_z_eta']=u[0,2]
        rec['model_z_bed']=u[0,3]

        Usurf=surf_interpolator(run_i,t)
        u=Usurf(x_samp)
        u=np.where(np.isfinite(u),u,0)
        # central differencing
        w=(u[2,1]-u[1,1])/(2*eps) - (u[4,0]-u[3,0])/(2*eps)

        rec['model_vor_surf']=w
        rec['model_u_surf']=u[0,0]
        rec['model_v_surf']=u[0,1]

        new_records.append(rec)
    
    return new_records
        
new_records=get_new_records()

model_data=pd.DataFrame(new_records).set_index('index')

assert np.all( segments.index.values == model_data.index.values )

joined=pd.merge(segments,model_data,left_index=True,right_index=True)

##
if 1: # quiver with all stations
    fig=plt.figure(2)
    fig.clf()
    ax=fig.add_axes([0,0,1,1])
    mod.grid.plot_edges(lw=0.5,ax=ax,color='0.75',zorder=-3)
    plt.setp(ax.get_xticklabels(),visible=0)
    plt.setp(ax.get_yticklabels(),visible=0)

    color_by='|u|'
    # color_by='Time (d)'
    
    if color_by=='|u|':
        scal=utils.mag(np.c_[joined.model_u.values,joined.model_v.values])
    else:
        scal=(joined.time.values-joined.time.values.min())/np.timedelta64(1,'D')
    quiv=ax.quiver(joined.xm.values,
                   joined.ym.values,
                   joined.model_u.values,
                   joined.model_v.values,
                   scal,
                   cmap='jet',
                   scale=30.0)
    plt.colorbar(quiv,label=color_by)
    ax.quiverkey(quiv,0.1,0.1,0.5,"0.5 m/s - depth avg")
    ax.axis( (647091.6894404562, 647486.7958566407, 4185689.605777047, 4185968.1328291544) )
    fig.savefig('model-extracted-velocities.png',dpi=200)

if 1: # scatter with vorticity
    fig=plt.figure(2)
    fig.clf()
    ax=fig.add_axes([0,0,1,1])
    mod.grid.plot_edges(lw=0.5,ax=ax,color='0.75',zorder=-3)
    plt.setp(ax.get_xticklabels(),visible=0)
    plt.setp(ax.get_yticklabels(),visible=0)

    scat=ax.scatter(joined.xm.values,
                    joined.ym.values,
                    15,joined['model_vor'],
                    cmap='PuOr')
    scat.set_clim([-0.05,0.05])
    plt.colorbar(scat,label="Vorticity, depth avg (s$^{-1}$)")
    ax.axis( (647091.6894404562, 647486.7958566407, 4185689.605777047, 4185968.1328291544) )
    fig.savefig('model-extracted-vorticity.png',dpi=200)

# surface plots

if 1: # quiver with all stations
    fig=plt.figure(2)
    fig.clf()
    ax=fig.add_axes([0,0,1,1])
    mod.grid.plot_edges(lw=0.5,ax=ax,color='0.75',zorder=-3)
    plt.setp(ax.get_xticklabels(),visible=0)
    plt.setp(ax.get_yticklabels(),visible=0)

    color_by='|u|'
    # color_by='Time (d)'
    if color_by=='|u|':
        scal=utils.mag(np.c_[joined.model_u_surf.values,
                             joined.model_v_surf.values])
    else:
        scal=(joined.time.values-joined.time.values.min())/np.timedelta64(1,'D')

    quiv=ax.quiver(joined.xm.values,
                   joined.ym.values,
                   joined.model_u_surf.values,
                   joined.model_v_surf.values,
                   scal,
                   cmap='jet',
                   scale=30.0)
    plt.colorbar(quiv,label=color_by)
    ax.quiverkey(quiv,0.3,0.1,0.5,"0.5 m/s - surface")
    ax.axis( (647091.6894404562, 647486.7958566407, 4185689.605777047, 4185968.1328291544) )
    fig.savefig('model-extracted-surf_velocities.png',dpi=200)

if 1: # scatter with vorticity
    fig=plt.figure(2)
    fig.clf()
    ax=fig.add_axes([0,0,1,1])
    mod.grid.plot_edges(lw=0.5,ax=ax,color='0.75',zorder=-3)
    plt.setp(ax.get_xticklabels(),visible=0)
    plt.setp(ax.get_yticklabels(),visible=0)

    scat=ax.scatter(joined.xm.values,
                    joined.ym.values,
                    15,joined['model_vor_surf'],
                    cmap='PuOr')
    scat.set_clim([-0.05,0.05])
    plt.colorbar(scat,label="Vorticity, surface (s$^{-1}$)")
    ax.axis( (647091.6894404562, 647486.7958566407, 4185689.605777047, 4185968.1328291544) )
    fig.savefig('model-extracted-surf_vorticity.png',dpi=200)
    
##

out_fn=inp_fn.replace(".csv","-model.csv")
assert out_fn!=inp_fn

joined.to_csv(out_fn,index=False)

