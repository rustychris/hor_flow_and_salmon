"""
Driver script for Suntans Head of Old River runs
"""
import six
import logging
log=logging.getLogger('hor_sun')
log.setLevel(logging.INFO)

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from stompy import filters, utils
from stompy.spatial import wkb2shp
from stompy.grid import unstructured_grid

from stompy.model.suntans import sun_driver as drv
##

import stompy.model.delft.dflow_model as dfm
six.moves.reload_module(utils)
six.moves.reload_module(dfm)
six.moves.reload_module(unstructured_grid)
six.moves.reload_module(drv)

model=drv.SuntansModel()
model.projection="EPSG:26910"

model.num_procs=4
model.sun_bin_dir="/home/rusty/src/suntans/main"
model.mpi_bin_dir="/usr/bin"

model.load_template("sun-template.dat")

# steady_000: IC is flat
# steady_001: IC includes bathy?
# steady_002: various changes to suntans, and a modified grid
#      to get rid of some bad cell center spacing
# steady_003: coarser triangle grid
# steady_004: 3D, nkmax=20
# steady_005: 3D, dt=1, nkmax=50
# steady_006: hex grid.
# steady_007: hex grid with improved centers - accidentally overwrote
# steady_008: down to dt=0.5 (longer run than 007)
model.set_run_dir(os.path.join('runs',"steady_008"),
                  mode='pristine')

model.run_start=np.datetime64('2017-03-01 00:00:00')
model.run_stop=np.datetime64('2017-03-01 06:00:00')

# model.set_grid(unstructured_grid.UnstructuredGrid.from_ugrid("../dfm/grid/derived/grid_bathy_ugrid.nc"))
# model.set_grid(unstructured_grid.UnstructuredGrid.from_ugrid("../grid/junction-grid-34-bathy.nc"))
# model.set_grid(unstructured_grid.UnstructuredGrid.from_ugrid("../grid/junction-grid-100-bathy.nc"))
# g=unstructured_grid.UnstructuredGrid.from_ugrid("../grid/junction-grid-101-bathy.nc")
# ds=xr.open_dataset("../grid/junction-grid-101-bathy.nc")
model.set_grid(unstructured_grid.UnstructuredGrid.from_ugrid("../grid/junction-grid-101-bathy.nc"))

# Vertical layers:
# freesurface BC is 2.2 (plus offset)
#  there is that deep hole down to -10m.
# what are the options for specifying layers?
# rstretch?
#

# Annoying, but suntans doesn't like signed elevations
# this offset will be applied to grid depths and freesurface boundary conditions.
model.z_offset=-10

model.config['maxFaces']=10

model.add_gazetteer("gis/forcing-v00.shp")

Q_upstream=drv.FlowBC(name='SJ_upstream',Q=210.0)
Q_downstream=drv.FlowBC(name='SJ_downstream',Q=-100.0)
h_old_river=drv.StageBC(name='Old_River',z=2.2)


model.add_bcs([Q_upstream,Q_downstream,h_old_river])

model.write()
model.partition()

##
# bc_sample=xr.open_dataset("/home/rusty/src/suntans/examples/EstuaryNetcdf/rundata/Estuary_BC.nc")
# ic_sample=xr.open_dataset("/home/rusty/src/suntans/examples/EstuaryNetcdf/rundata/Estuary_IC.nc")
# met_sample=xr.open_dataset("/home/rusty/src/suntans/examples/EstuaryNetcdf/rundata/Estuary_MetForcing.nc")

# model.partition() # not yet
# model.run_model()
# $(MPIHOME)/bin/mpirun -np $(NUMPROCS) ./$(EXEC) -g -s -vv --datadir=$(datadir)

##

# it's getting twitchy at a narrow triangle.
# would it help to adjust circumcenters?
# This currently occurs at 34s into the run.
# any different with CorrectVoronoi=1?
# that completes this short run.
# There is a twitch at the beginning which is limiting an increase
# in the timestep.

# It may be enough to alter the initial condition to set eta to max(dv,bc_eta)

# fixdzz=0 in suntans.dat seems to help
# dt=0.2 still unstable.
# looks like the problem arises on edges between wet and dry.
# but in these cells they appear bone dry the entire time.
#   maybe the code is too strict on what qualifies as dry?
#   SMALL
#   #define DRYCELLHEIGHT 1e-10

# try skipping netcdf IC altogether.
# same issues, it appears.
# changing drycellheight to 1e-3, I get significant velocities
# all over the place on the levees.
# almost seems like one part of the code props up fs to drycellheight,
# and somewhere else it interprets it then as a wet cell?

# What is bufferheight?  an edge depth below which non-dry edges
#  get a high CdB.
# when are cells and edges set as inactive in the modern phys.c?
#   reading netcdf initial freesurface avoids one chance to prop up
#   the freesurface.
# there is some funny business in SetFluxHeight which claims to be assuming
# central differencing??

# would be nice to see how dzf and etop, etc. are being set.
# if mergeArrays is set to 0 maybe I get that output??
# yes.

# more places to fix in the grid?
#  Location: x=6.488e+05, y=4.184e+06, z=-8.501e+00
# 0.25m grid spacing.

# It's running now, but seems like maybe none of the BCs are active?
##

