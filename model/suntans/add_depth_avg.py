import xarray as xr
import sys
import os

for fn in sys.argv[1:]:
    print(fn)

    ds=xr.open_dataset(fn)

    ds['uavg']=(ds.uc*ds.dzz).sum(dim='Nk') / ds.dzz.sum(dim='Nk')
    ds['vavg']=(ds.vc*ds.dzz).sum(dim='Nk') / ds.dzz.sum(dim='Nk')

    fn_out=fn.replace('.nc','-avg.nc')
    assert fn_out!=fn
    assert not os.path.exists(fn_out),"Output %s exists"%fn_out
    ds.to_netcdf(fn_out)



