from __future__ import print_function

import sys
import os
import numpy as np

import stompy.model.delft.io as dio
from stompy.spatial import wkb2shp

shp_fn,run_dir = sys.argv[1:]

##

forcing=wkb2shp.shp2geom(shp_fn)

print("shp_fn: %s"%shp_fn)
print("run_dir: %s"%run_dir)

for f in forcing:
    print( f['name'] )
    pli_fn=os.path.join(run_dir,"%s.pli"%f['name'])

    with open(pli_fn,'wt') as fp:
        dio.write_pli(fp, [ [f['name'], np.array( f['geom'].coords )] ] )
