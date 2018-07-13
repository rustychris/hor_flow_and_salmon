"""
Apply the interpolation method of Zhang 2016 to this model domain.
"""
import xarray as xr
from stompy.grid import unstructured_grid
from stompy.model.delft import dfm_grid

##

map_fn="../model/dfm/dfm/runs/hor_002/DFM_OUTPUT_flowfm/flowfm_0003_20120801_merged_map.nc"

ds=xr.open_dataset(map_fn)

##

six.moves.reload_module(dfm_grid)

g=dfm_grid.DFMGrid(ds)

##

