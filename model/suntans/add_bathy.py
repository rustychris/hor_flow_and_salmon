"""
Add bathy data to grid.
Had been in the grid directory, but moved here for quicker edit-run cycles
"""
from stompy import utils
from stompy.grid import unstructured_grid
import numpy as np

utils.path("../../bathy")
import bathy

def add_bathy(g):
    dem=bathy.dem()

    def eval_pnts(x):
        z=dem(x)
        z[np.isnan(z)]=5.0
        return z

    # For this high resolution grid don't worry so much
    # about averaging or getting clever about bathymetry
    g.add_node_field('node_depth', eval_pnts(g.nodes['x']), on_exists='overwrite')
    g.add_edge_field('edge_depth', eval_pnts(g.edges_center()),on_exists='overwrite')
    g.add_cell_field('cell_depth', eval_pnts(g.cells_center()),on_exists='overwrite')
    g.add_cell_field('depth', g.cells['cell_depth'])

    return g

