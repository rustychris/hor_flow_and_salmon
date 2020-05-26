import os
from stompy.spatial import field
from stompy import memoize

@memoize.memoize()
def dem(suffix=''):
    if suffix=='':
        return field.GdalGrid(os.path.join(os.path.dirname(__file__),
                                           'junction-composite-20190117-no_adcp.tif'))
    elif suffix=='adcp':
        return field.GdalGrid(os.path.join(os.path.dirname(__file__),
                                           'junction-composite-dem.tif'))
    else:
        raise Exception("Unknown bathy suffix/version: %s"%suffix)
        
    #return field.MultiRasterField(['/home/rusty/data/bathy_dwr/gtiff/*.tif'])


