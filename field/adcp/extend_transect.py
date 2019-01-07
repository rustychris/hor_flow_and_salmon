"""
Dev: extrapolate ADCP velocity field to surface and bed. 
"""
import six
from stompy import xr_transect
six.moves.reload_module(xr_transect)

## 

tran=xr.open_dataset('040518_BT/040518_5BTref-avg.nc')

##
xr_transect.add_rozovski(tran)

##

plt.figure(1).clf()
xr_transect.plot_scalar(tran,tran.Uroz.sel(roz='downstream'))

plt.plot(tran.d_sample,z_sgn*tran.z_bed,'k-',lw=0.5)

##

six.moves.reload_module(xr_transect)

var_methods=[ ('Uroz',dict(roz=0),'linear','constant'),
              ('Uroz',dict(roz=1),'linear','constant'),
              ('U',dict(xy=0),'linear','constant'),
              ('U',dict(xy=1),'linear','constant') ]

ds=xr_transect.extrapolate_vertical(tran,var_methods,eta=0)

## 
plt.figure(1).clf()
xr_transect.plot_scalar(ds,ds.Uroz.sel(roz='downstream'))

plt.plot(ds.d_sample,z_sgn*ds.z_bed,'k-',lw=0.5)

##

# This yields 214m3/s, compared to 164m3/s without the extrapolation.
xr_transect.Qleft(ds)
