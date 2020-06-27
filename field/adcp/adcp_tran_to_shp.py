import glob
from stompy.memoize import memoize
import read_sontek

##

# Data from each set of repeats is in a single folder
transects=glob.glob("040518_BT/*BTref")

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

@memoize(lru=10)
def read_transect(transect):
    """
    Find and load transects from a folder, calling tweak_sontek on each.
    return list of xr.Dataset
    """
    rivr_fns=glob.glob('%s/*.rivr'%transect) + glob.glob('%s/*.riv'%transect)

    tran_dss=[ tweak_sontek(read_sontek.surveyor_to_xr(fn,proj='EPSG:26910'))
               for fn in rivr_fns ]
    return tran_dss


##
from shapely import geometry
from collections import defaultdict

geoms=[]
fields=defaultdict(list)

for t in transects:
    dss=read_transect(t)
    for ds in dss:
        geom=geometry.LineString( np.c_[ds.x_sample,ds.y_sample] )
        geoms.append(geom)
        fields['dir'].append(t)

##

from stompy.spatial import wkb2shp

wkb2shp.wkb2shp('transects_2018.shp',geoms,overwrite=True,
                fields=fields)
