from stompy import utils
from stompy.grid import unstructured_grid
import numpy as np

utils.path("../../bathy")
import bathy

grid_out="junction-grid-203-bathy.nc"
#grid_out="junction-grid-101-bathy.nc"

#g=unstructured_grid.SuntansGrid('junction-grid-100')
#g=unstructured_grid.UnstructuredGrid.read_suntans_hybrid('junction-grid-101')
#g=unstructured_grid.UnstructuredGrid.read_suntans_hybrid('junction-grid-202')
g=unstructured_grid.UnstructuredGrid.from_ugrid('full_merge_v15/edit08.nc')

dem=bathy.dem()

def eval_pnts(x):
    z=dem(x)
    z[np.isnan(z)]=5.0
    return z

# For this high resolution grid don't worry so much
# about averaging or getting clever about bathymetry
g.add_node_field('depth', eval_pnts(g.nodes['x']))
g.add_edge_field('edge_depth', eval_pnts(g.edges_center()))
g.add_cell_field('cell_depth', eval_pnts(g.cells_center()))


g.write_ugrid(grid_out,overwrite=True)

