import matplotlib
matplotlib.use('Agg')

from stompy.model.suntans import sun_driver
from stompy import utils
from stompy.spatial import wkb2shp
import matplotlib.pyplot as plt
import six
import numpy as np
import xarray as xr
import os

try:
    base=os.path.dirname(__file__)
except NameError:
    base="/home/rusty/src/hor_flow_and_salmon/model/suntans"

utils.path(os.path.join(base,"../../field/adcp"))
from stompy import xr_transect
import summarize_xr_transects

##
# plot a few a sections
# Run from the run directory
run_dir="."
model=sun_driver.SuntansModel.load(run_dir)

tran_shp=os.path.join(base,"../../gis/model_transects.shp")
tran_geoms=wkb2shp.shp2geom(tran_shp)

fig_dir=os.path.join(model.run_dir,'figs-20181030')
os.path.exists(fig_dir) or os.makedirs(fig_dir)

for ti,t in enumerate(tran_geoms):
    print(t['name'])
    # if t['name']!='2018_07': continue # DBG

    if t['name']=='2018_09':
        obs=xr.open_dataset(os.path.join(base,"../../field/adcp/040518_BT/040518_9BTref-avg.nc"))
    elif t['name']=='2018_08':
        obs=xr.open_dataset(os.path.join(base,"../../field/adcp/040518_BT/040518_8BTref-avg.nc"))
    elif t['name']=='2018_07':
        obs=xr.open_dataset(os.path.join(base,"../../field/adcp/040518_BT/040518_7_BTref-avg.nc"))
    elif t['name']=='2018_06':
        obs=xr.open_dataset(os.path.join(base,"../../field/adcp/040518_BT/040518_6BTref-avg.nc"))
    elif t['name']=='2018_05':
        obs=xr.open_dataset(os.path.join(base,"../../field/adcp/040518_BT/040518_5BTref-avg.nc"))
    elif t['name']=='2018_04':
        obs=xr.open_dataset(os.path.join(base,"../../field/adcp/040518_BT/040518_4BTref-avg.nc"))
    elif t['name']=='2018_03':
        obs=xr.open_dataset(os.path.join(base,"../../field/adcp/040518_BT/040518_3_BTref-avg.nc"))
    elif t['name']=='2018_02':
        obs=xr.open_dataset(os.path.join(base,"../../field/adcp/040518_BT/040518_2BTref-avg.nc"))
    elif t['name']=='2018_01':
        obs=xr.open_dataset(os.path.join(base,"../../field/adcp/040518_BT/040318_1_BTref-avg.nc"))
    else:
        obs=None

    if obs is not None:
        print("Pulling transect geometry from line fit to observations")
        xy=np.c_[ obs.orig_x_sample.values, obs.orig_y_sample.values ]
        xy=xy[np.isfinite(xy[:,0]),:]
    else:
        print("Pulling transect geometry from shapefile")
        xy=np.array(t['geom'])

    tran=model.extract_transect(xy=xy,time=-1,dx=2)
    if tran is None:
        print("No overlap for %s, moving on"%t['name'])
        continue

    if tran.attrs['source']=='.':
        tran.attrs['source']=os.path.basename(os.getcwd())

    wet_samples=np.nonzero(np.abs(tran.z_dz.sum(dim='layer').values)>0.01)[0]
    sample_slice=slice(wet_samples[0],wet_samples[-1]+1)
    tran=tran.isel(sample=sample_slice)
    tran.attrs['transect_name']=t['name']
    tran.to_netcdf( os.path.join(fig_dir,"transect-%s.nc"%t['name']) )
    fig=summarize_xr_transects.summarize_transect(tran,num=20+ti,w_scale=0.0,plot_averages=True)

    if obs is not None:
        mod_seg=np.c_[ tran.x_sample.values, tran.y_sample.values ]
        obs_res=xr_transect.resample_to_common([obs],resample_z=False,seg=mod_seg)[0]
        xr_transect.add_rozovski(obs_res)
        Uavg=xr_transect.depth_avg(obs_res, obs_res.Uroz.isel(roz=0))
        fig.axes[0].plot(obs_res.d_sample,Uavg,color='orange',label='Obs')
        umax=Uavg.max()
        if umax>fig.axes[0].axis()[3]:
            fig.axes[0].axis(ymax=umax)
        # bathy comparison
        fig.axes[3].plot(obs_res.d_sample, tran.eta.mean()-obs_res.z_bed, 'k--',label='Obs. bed')

    fig.savefig(os.path.join(fig_dir,"transect-%s.png"%t['name']))





