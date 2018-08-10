from stompy import utils
from stompy.grid import unstructured_grid

utils.path("../../bathy")
import bathy

#grid_in="junction-grid-34.nc"
grid_out="junction-grid-34-bathy.nc"

g=unstructured_grid.UnstructuredGrid.read_suntans_hybrid('junction-grid-34')

dem=bathy.dem()

# For this high resolution grid don't worry so much
# about averaging or getting clever about bathymetry
g.add_node_field('depth', dem(g.nodes['x']))
g.add_edge_field('edge_depth', dem(g.edges_center()))
g.add_cell_field('cell_depth', dem(g.cells_center()))

g.write_ugrid(grid_out,overwrite=True)
