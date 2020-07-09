from stompy import utils
utils.path("/home/rusty/src")
from soda.dataio.ugrid import suntans2untrim
from soda.dataio.suntans import sunpy

import six
six.moves.reload_module(sunpy)
six.moves.reload_module(suntans2untrim)

# DEBUGGING
ncfile = 'runs/short045_g8.24_20180405/average.nc_0000.nc'
outfile = 'runs/short045_g8.24_20180405/ptm_average.nc_0000.nc'

suntans2untrim.suntans2untrim(ncfile, outfile, None, None)
