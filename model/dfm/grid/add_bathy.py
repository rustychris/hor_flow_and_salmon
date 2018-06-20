import sys
from stompy import utils

from stompy.grid import unstructured_grid
utils.path("../../../bathy")
import bathy

def process(grid_in,grid_out):
    g=unstructured_grid.UnstructuredGrid.from_ugrid(grid_in)

    dem=bathy.dem()

    # By default, just put depths onto the nodes straight from DEM
    g.add_node_field('depth', dem(g.nodes['x']))

    g.write_ugrid(grid_out,overwrite=True)


if __name__=='__main__':
    process(sys.argv[1],sys.argv[2])
else:
    process('derived/grid_ugrid.nc','derived/grid_bathy_ugrid.nc')
