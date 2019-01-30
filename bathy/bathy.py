import os
from stompy.spatial import field
from stompy import memoize

@memoize.memoize()
def dem():
    #return field.MultiRasterField(['/home/rusty/data/bathy_dwr/gtiff/*.tif'])
    #return field.GdalGrid(os.path.join(os.path.dirname(__file__),
    #                                   'junction-composite-dem.tif'))
    return field.GdalGrid(os.path.join(os.path.dirname(__file__),
                                       'junction-composite-20190117-no_adcp.tif'))



