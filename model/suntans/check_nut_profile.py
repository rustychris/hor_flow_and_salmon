from stompy.model.delft import dflow_model
from stompy.model.suntans import sun_driver
from stompy import utils
from stompy.spatial import wkb2shp
import matplotlib.pyplot as plt
import six
import numpy as np
import xarray as xr
utils.path("../../field/adcp")
from stompy import xr_transect
import summarize_xr_transects

##
six.moves.reload_module(dflow_model)
six.moves.reload_module(sun_driver)

# plot a few a sections
run_dir="runs/steady_036"
model=sun_driver.SuntansModel.load(run_dir)

tran_shp="../../gis/model_transects.shp"
tran_geoms=wkb2shp.shp2geom(tran_shp)

for ti,t in enumerate(tran_geoms):
    if t['name'] not in ['2018_05']:
        continue # Focus on transect 5
    break

xy=np.array(t['geom'])
tran=model.extract_transect(xy=xy,time=-1,dx=2,
                            vars=['uc','vc','Ac','dv','dzz','eta','nu_v'])
wet_samples=np.nonzero(np.abs(tran.z_dz.sum(dim='layer').values)>0.01)[0]
sample_slice=slice(wet_samples[0],wet_samples[-1]+1)
tran=tran.isel(sample=sample_slice)

plt.figure(2).clf()
coll=xr_transect.plot_scalar(tran,tran.nu_v,cmap='jet')
plt.colorbar(coll,label=r'$\nu_T$')

