"""
Use the figures defined in summarize_xr_transects to plot up
the 2018 ADCP observations
"""
import read_sontek, summarize_xr_transects
import glob
import six
import numpy as np
import xarray as xr

from stompy import xr_transect

##
six.moves.reload_module(xr_transect)

def tweak_sontek(ds):
    """
    """
    if '20180404085311r.rivr' in ds.rivr_filename:
        print("%s is known to be offset along channel"%ds.rivr_filename)
    # that transect is also missing some lat/lon
    bad_sample=(ds.lon.values==0.0)
    if np.any(bad_sample):
        print("%d of %d missing lat/lon"%(bad_sample.sum(),len(bad_sample)))
        ds=ds.isel(sample=~bad_sample)
    return ds

# Data from each set of repeats is in a single folder
transect_dirs=glob.glob("040518_BT/*BTref")

for transect_dir in transect_dirs:
    avg_fn=transect_dir+'-avg.nc'
    if os.path.exists(avg_fn):
        continue
    rivr_fns=glob.glob('%s/*.rivr'%transect_dir) + glob.glob('%s/*.riv'%transect_dir)

    tran_dss=[ tweak_sontek(read_sontek.surveyor_to_xr(fn,proj='EPSG:26910'))
               for fn in rivr_fns ]

    ds=xr_transect.average_transects(tran_dss)

    ds['U']=('sample','layer','xy'), np.concatenate( (ds.Ve.values[:,:,None],
                                                      ds.Vn.values[:,:,None]), axis=2)

    if xr_transect.Qleft(ds)<0:
        ds=ds.isel(sample=slice(None,None,-1))
        ds.d_sample[:]=ds.d_sample[0]-ds.d_sample

    if 0: # Extend the transects in the vertical, too.
        var_methods=[ ('U',dict(xy=0),'linear','constant'),
                      ('U',dict(xy=1),'linear','constant') ]
        ds=xr_transect.extrapolate_vertical(tran,var_methods,eta=0)
        
    ds.attrs['source']=transect_dir

    ds.to_netcdf(avg_fn)

##
six.moves.reload_module(summarize_xr_transects)

extrap=True

tran_ncs=glob.glob(os.path.join( os.path.dirname(transect_dir), "*-avg.nc"))
for i,fn in enumerate(tran_ncs):
    # if '_5BTref' not in fn: continue # DBG
    print(fn)
    ds=xr.open_dataset(fn)

    if extrap:
        ds=xr_transect.extrapolate_vertical(ds,[ ('U',dict(xy=0),'linear','constant'),
                                                 ('U',dict(xy=1),'linear','constant') ],
                                            eta=0)

    fig=summarize_xr_transects.summarize_transect(ds,num=20+i,w_scale=0.0,plot_averages=True)
    ds.close()
    img_ext='.png'
    if extrap:
        img_ext='-z_extrap' + img_ext

    fig.savefig(fn.replace('.nc',img_ext))

