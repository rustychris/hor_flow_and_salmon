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

from stompy.model.delft import dflow_model
six.moves.reload_module(dflow_model)

class HORModel(dflow_model.DFlowModel):
    dfm_bin_dir="/home/rusty/src/dfm/r53925-opt/bin"
    nprocs=4
    z_datum='NAVD88'
    projection='EPSG:26910'

    def bc_factory(self,params):
        if params['name']=='SJ_upstream':
            return self.flow_bc(Q=170.0,**params)
        elif params['name']=='SJ_downstream':
            return self.stage_bc(z=2.0,**params)
        elif params['name']=='Old_River':
            return self.stage_bc(z=2.0,**params)
        else:
            raise Exception("Unrecognized %s"%str(params))

model=HORModel()

base_dir="."

# Parameters to control more specific aspects of the run
# test_24h: proof of concept,  all free surface forcing (2.1,2.0,2.0), Kmx=2
# hor_002: 4 cores, 10 layers, inflow, stage at outflow
model.run_name="hor_002"
model.run_start=np.datetime64('2012-08-01')
model.run_stop=np.datetime64('2012-08-02')

model.set_run_dir(os.path.join('runs',model.run_name), 'create')

model.set_grid("../grid/derived/grid_net.nc")

model.load_mdu('template.mdu')

model.set_cache_dir('cache')


# OPTION A:
#  Fields of the shapefile evaluate to objects which are added to the
#  list of boundary conditions.

# features which have manually set locations for this grid
model.add_bcs_from_shp(os.path.join(base_dir,'gis','forcing-v00.shp'))

model.write()

model.partition()


