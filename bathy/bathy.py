import os
from stompy.spatial import field
from stompy import memoize

@memoize.memoize()
def dem(suffix=''):
    if suffix=='':
        return field.GdalGrid(os.path.join(os.path.dirname(__file__),
                                           'junction-composite-20190117-no_adcp.tif'))
    elif suffix=='smooth':
        return field.GdalGrid(os.path.join(os.path.dirname(__file__),
                                           'junction-composite-20200603-w_smooth.tif'))
    elif suffix=='smoothadcp':
        return field.GdalGrid(os.path.join(os.path.dirname(__file__),
                                           'junction-composite-20200604-w_smooth.tif'))
    elif suffix=='smoothadcp2':
        # this includes adcp-derived bathy above the junction, while
        # the above uses dwr multibeam above the junction
        return field.GdalGrid(os.path.join(os.path.dirname(__file__),
                                           'junction-composite-20200605-w_smooth.tif'))
    elif suffix=='adcp':
        return field.GdalGrid(os.path.join(os.path.dirname(__file__),
                                           'junction-composite-dem.tif'))
    else:
        raise Exception("Unknown bathy suffix/version: %s"%suffix)


