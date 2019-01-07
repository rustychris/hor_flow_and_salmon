"""
Render roughness map
"""
import numpy as np
from stompy.spatial import field
from stompy.spatial import interp_coverage

import os

def factory(attrs):
    geo_bounds=attrs['geom'].bounds

    if attrs['src_name'].startswith('py:'):
        expr=attrs['src_name'][3:]
        # something like 'ConstantField(-1.0)'
        # a little sneaky... make it look like it's running
        # after a "from stompy.spatial.field import *"
        # and also it gets fields of the shapefile
        field_hash=dict(field.__dict__)
        # convert the attrs into a dict suitable for passing to eval
        attrs_dict={}
        for name in attrs.dtype.names:
            attrs_dict[name]=attrs[name]
        return eval(expr,field_hash,attrs_dict)

    assert False

src_shp='roughness_sources.shp'

mbf=field.CompositeField(shp_fn=src_shp,
                         factory=factory,
                         priority_field='priority',
                         data_mode='data_mode',
                         alpha_mode='alpha_mode')

xxyy=[642800, 649400, 4183200, 4188400]
bleed=50
xxyy_pad=[ xxyy[0]-bleed,
           xxyy[1]+bleed,
           xxyy[2]-bleed,
           xxyy[3]+bleed ]
dem=mbf.to_grid(dx=2,dy=2,bounds=xxyy_pad)

dem=dem.crop(xxyy)
out_fn='composite-roughness.tif'
os.path.exists(out_fn) and os.unlink(out_fn)
dem.write_gdal(out_fn)

