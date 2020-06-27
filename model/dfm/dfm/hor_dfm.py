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

model=dflow_model.DFlowModel()

model.dfm_bin_dir=os.path.join(os.environ['HOME'],"src/dfm/r53925-opt/bin")
model.mpi_bin_dir="/opt/cse/bin"


model.num_procs=4
model.z_datum='NAVD88'
model.projection='EPSG:26910'

# Parameters to control more specific aspects of the run
# test_24h: proof of concept,  all free surface forcing (2.1,2.0,2.0), Kmx=2
# hor_002: 4 cores, 10 layers, inflow, stage at outflow
# hor_003: adjust BCs:
#          SJ inflow to 210, based on average of River Surveyor-reported flows at 2B.
#          drop water level on OR to 1.50m and SJ to 1.55m.  This should put a bit
#          more flow onto the OR side to get better split w.r.t. ADCP.  Dropping
#          the free surface should help increase velocities, as the increase flow rate
#          is probably not enough alone.
# hor_004: change SJ downstream to outflow condition
#          reduce baseline vertical eddy viscosity, reduce horizontal eddy viscosity
#          switch to gazetteer interface
model.set_run_dir("runs/hor_004", mode='clean')

model.run_start=np.datetime64('2012-08-01')
model.run_stop=np.datetime64('2012-08-02')

model.set_grid("../grid/derived/grid_net.nc")

model.load_mdu('template.mdu')

model.set_cache_dir('cache')

model.add_gazetteer('gis/forcing-v00.shp')

model.add_flow_bc(Q=210,name='SJ_upstream')
model.add_flow_bc(Q=-105,name='SJ_downstream')
model.add_stage_bc(z=1.50,name='Old_River')

if __name__=='__main__':
    model.write()
    print("partitioning")
    model.partition()
    print("Running")
    model.run_model()




