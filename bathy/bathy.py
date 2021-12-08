import os
from stompy.spatial import field
from stompy import memoize

here=os.path.dirname(__file__)

@memoize.memoize()
def dem(suffix=''):
    # This DEM covers a larger area, but doesn't have the extra processing
    # around the junction
    base_fn=os.path.join(here,"junction-composite-dem.tif")

    if suffix=='':
        jct_fn=os.path.join(here,'junction-composite-20190117-no_adcp.tif')
    elif suffix=='smooth':
        jct_fn=os.path.join(here,'junction-composite-20200603-w_smooth.tif')
    elif suffix=='smoothadcp':
        jct_fn=os.path.join(here,'junction-composite-20200604-w_smooth.tif')
    elif suffix=='smoothadcp2':
        # this includes adcp-derived bathy above the junction, while
        # the above uses dwr multibeam above the junction
        jct_fn=os.path.join(here,'junction-composite-20200605-w_smooth.tif')
    elif suffix=='adcp':
        jct_fn=None
    else:
        raise Exception("Unknown bathy suffix/version: %s"%suffix)

    if jct_fn is not None:
        # Specify priority to be sure jct fn wins out
        f=field.MultiRasterField([(jct_fn,10),
                                  (base_fn,0)])
    else:
        f=field.GdalGrid(base_fn)
        
    return f


