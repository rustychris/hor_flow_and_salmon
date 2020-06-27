# Towards a continuous notion of spatial dependence.

# define an anisotropy field as a diffusion tensor
# as a function of space.
# each sample then diffuses according to its diffusion tensor.
#
# then either follow the approach of the grid diffusion solver where fluxes
# allow for weighting of samples, or weight all samples equally.

# numerical diffusion would be a problem, though we can use particle tracking
# through the diffusion field to deal with that.

# This will have the same max/min property issues as IDW, though.

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from stompy import utils
from stompy.grid import unstructured_grid
from stompy.model.delft import dfm_grid
from stompy.spatial import wkb2shp, proj_utils, field
import stompy.plot.cmap as scmap
cmap=scmap.load_gradient('hot_desaturated.cpt')
six.moves.reload_module(unstructured_grid)

##
import bathy
dem=bathy.dem()
##
adcp_shp=wkb2shp.shp2geom('derived/samples-depth.shp')
adcp_ll=np.array( [ np.array(pnt) for pnt in adcp_shp['geom']] )
adcp_xy=proj_utils.mapper('WGS84','EPSG:26910')(adcp_ll)
adcp_xyz=np.c_[ adcp_xy,adcp_shp['depth'] ]
##
# Rather than use the ADCP data directly, during testing
# use its horizontal distribution, but pull "truth" from the
# DEM
xyz_input=adcp_xyz.copy()
xyz_input[:,2] = dem( xyz_input[:,:2] )
##

clip=(646900, 647600, 4185500, 4186000)
tile=dem.extract_tile(clip)

##

dx=5
dy=5

Nx=(clip[1]-clip[0])//dx
Ny=(clip[3]-clip[2])//dy

extents=clip




