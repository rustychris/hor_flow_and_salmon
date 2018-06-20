from stompy.spatial import field
from stompy import memoize

@memoize.memoize()
def dem():
    return field.MultiRasterField(['/home/rusty/data/bathy_dwr/gtiff/*.tif'])



