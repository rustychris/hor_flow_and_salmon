import xarray as xr
from stompy import utils

##

ds=xr.open_dataset("runs/snubby_cfg001_20180316/Estuary_BC.nc")
ds=ds.set_coords('time')

##
da=ds.boundary_Q.isel(Nseg=0)

## 
# This is just shoving zeros in there.
six.moves.reload_module(utils)
da_fill=utils.fill_tidal_data(da,fill_time=False)

##

plt.figure(1).clf()
plt.plot( da_fill.time, da_fill )
