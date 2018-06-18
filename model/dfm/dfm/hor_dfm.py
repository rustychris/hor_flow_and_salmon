#!/usr/bin/env python
"""
Script to configure and generate mdu ready for running dflow fm.

"""

import os
import glob
import pdb
import io
import shutil
import subprocess
import numpy as np
import logging
import xarray as xr
import six

from stompy import utils
import stompy.model.delft.io as dio
from stompy.model.delft import dfm_grid
from stompy.grid import unstructured_grid
from stompy.spatial import wkb2shp
from stompy.io.local import usgs_nwis,noaa_coops

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)

log=logging.getLogger('hor_dfm')


## --------------------------------------------------

base_dir="."
dfm_bin_dir="/Users/rusty/src/dfm/r52184-opt/bin"
nprocs=1


# Parameters to control more specific aspects of the run
if 1: # nice short setup for testing:
    run_name="test_24h" 
    run_start=np.datetime64('2012-08-01')
    run_stop=np.datetime64('2012-08-02')


run_dir=os.path.join(base_dir,'runs',run_name)

grid=dfm_grid.DFMGrid("../grid/derived/grid_net.nc")

##

# Make sure run directory exists:
os.path.exists(run_dir) or os.makedirs(run_dir)

## --------------------------------------------------------------------------------
# Edits to the template mdu:
# 

mdu=dio.MDUFile('template.mdu')

mdu.set_time_range(start=run_start,stop=run_stop)

mdu.set_filename(os.path.join(run_dir,'flowfm.mdu'))

##

bc_fn=mdu.filepath(['external forcing','ExtForceFile'])

# clear any stale bc files:
for fn in [bc_fn]:
    os.path.exists(fn) and os.unlink(fn)

##

import stompy.model.delft.dfm_bc as bc

six.moves.reload_module(bc)

bc.BC.set_cache_dir('cache',create=1)

# features which have manually set locations for this grid
forcing_shp=os.path.join(base_dir,'gis','forcing-v00.shp')
features=wkb2shp.shp2geom(forcing_shp)

    
##

# Copy grid file into run directory and update mdu
mdu['geometry','NetFile'] = os.path.basename(grid.filename)
dest=os.path.join(run_base_dir, mdu['geometry','NetFile'])
dfm_grid.write_dfm(grid,dest,overwrite=True)


mdu.write()

##

mdu.partition(nprocs,dfm_bin_dir=dfm_bin_dir)

