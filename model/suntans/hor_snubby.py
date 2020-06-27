"""
Driver script for Suntans Head of Old River runs.

This version uses the snubby grid
"""
import six
import logging
log=logging.getLogger('hor_sun')
log.setLevel(logging.INFO)

import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from stompy import filters, utils
from stompy.spatial import wkb2shp, field
from stompy.grid import unstructured_grid

from stompy.model.suntans import sun_driver as drv
import stompy.model.delft.dflow_model as dfm
import add_bathy
import grid_roughness

##

with open('local_config.py') as fp:
    exec(fp.read())

##

model=drv.SuntansModel()
model.projection="EPSG:26910"

model.num_procs=4
model.sun_bin_dir="/home/rusty/src/suntans/main"
model.mpi_bin_dir="/usr/bin"

model.load_template("sun-template.dat")

# Suite of runs to reassess status:
# all using parabolic.
# steady_038: 2D, z0B=0.001
# steady_039: 3D, z0B=0.001, with advection.
# steady_040: 3D, z0B=0.001, no advection
# steady_041: 2D no advection.
# snubby_042: 2D, transition to snubby grid.
# snubby_043:    raise H bc form
# snubby_044:    drop Q to 220. that was still nonlinear=0
# snubby_045:  doubled grid. nonlinear=0
# snubby_046:  doubled grid. nonlinear=1
# snubby_047: original grid, nonlinear=1
# snubby_048:  nonlinear=5 (TVD)
# snubby_049:  nonlinear=1, shuffle edges - no difference
# snubby_050: smooth the bathymetry
# snubby_051: more extreme smoothing.
# snubby_052: ditch smoothing, try central difference.
#        053: less Kh, crashes
# snubby_054:  nonlinear=1, but suntans is ditching stmp from advection
#      that confirms that stmp from nonlinear is the culprit.
# snubby_055: change up the stmp output so it is *only* the horizontal advection.
# snubby_056: single processor run
# snubby_057: towards variable roughness, z0B=0.01*bathy_rms
#        058: not enough effect, go to z0B=0.1*bathy_rms
#        059: fine, go to z0B=bathy_rms
#        060: good improvement, now return to 3D.
#        061: cleanup, and try real turbulence not parabolic.
#        062: record CFL Limiting cells
#        063: no dredging.
#        064: smoother barrier in the DEM
#        065: tweak partitioning parameters, and try parabolic
#        066: keep parabolic, double roughness
#        067: half roughness (relative to 065)
#        068: stick with z0B=0.5*bathy_rms, and try half the vertical layers
model.set_run_dir(os.path.join('runs',"snubby_068"),
                  mode='askclobber')

model.run_start=np.datetime64('2017-03-01 00:00:00')
model.run_stop=np.datetime64('2017-03-01 04:00:00')
ramp_hours=1 # how quickly to increase the outflow

model.config['Nkmax']=50

dt=0.5
model.config['dt']=dt
model.config['ntout']=int(1800./dt)
model.config['ntoutStore']=int(3600./dt)

model.config['nonlinear']=1
if int(model.config['Nkmax'])==1:
    model.config['stairstep']=0 # 2D, partial step
else:
    model.config['stairstep']=1 # 3D, stairstep
model.config['thetaM']=-1
model.config['z0B']=0.001
model.config['nu_H']=0.0
model.config['nu']=1e-5
model.config['turbmodel']=10 # 1: my25, 10: parabolic
model.config['CdW']=0.0
model.config['wetdry']=1
model.config['maxFaces']=4

grid_src="../grid/snubby_junction/snubby-01.nc"
grid_bathy="snubby-with_bathy.nc"

import add_bathy
import grid_roughness

if not os.path.exists(grid_bathy) or os.stat(grid_bathy).st_mtime < os.stat(grid_src).st_mtime:
    g_src=unstructured_grid.UnstructuredGrid.from_ugrid(grid_src)
    add_bathy.add_bathy(g_src)
    grid_roughness.add_roughness(g_src)
    g_src.write_ugrid(grid_bathy,overwrite=True)

g=unstructured_grid.UnstructuredGrid.from_ugrid(grid_bathy)
g.orient_edges()

## 

model.set_grid(g)

model.grid.modify_max_sides(4)

## 
# Vertical layers:
# freesurface BC is 2.2 (plus offset)
#  there is that deep hole down to -10m.
# what are the options for specifying layers?
# rstretch?
#

# Annoying, but suntans doesn't like signed elevations
# this offset will be applied to grid depths and freesurface boundary conditions.
# this was -10, but that leaves a lot of dead space on top.
model.z_offset=-4

model.config['maxFaces']=4

model.add_gazetteer("../grid/snubby_junction/forcing-snubby-01.shp")

Q_upstream=drv.FlowBC(name='SJ_upstream',Q=220.0,dredge_depth=None)
Qdown=-100 # target outflow
# ramp up to the full outflow over 1h
h=np.timedelta64(1,'h')
Qdown=xr.DataArray( data=[0,0,Qdown,Qdown,Qdown],name='Q',
                    dims=['time'],
                    coords=dict(time=[model.run_start-24*h,
                                      model.run_start,
                                      model.run_start+ramp_hours*h,
                                      model.run_stop,
                                      model.run_stop+24*h]) )
Q_downstream=drv.FlowBC(name='SJ_downstream',Q=Qdown,dredge_depth=None)
# 2.5 crashed.
# h_old_river=drv.StageBC(name='Old_River',z=2.2-0.65)
h_old_river=drv.StageBC(name='Old_River',z=2.5)

model.add_bcs([Q_upstream,Q_downstream,h_old_river])


## 
model.write()

# bathy rms ranges from 0.015 to 1.5
cell_z0B=0.5*model.grid.cells['bathy_rms']
e2c=model.grid.edge_to_cells()
nc1=e2c[:,0]
nc2=e2c[:,1]
nc2[nc2<0]=nc1[nc2<0]
edge_z0B=0.5*( cell_z0B[nc1] + cell_z0B[nc2] )
model.ic_ds['z0B']=('time','Ne'), edge_z0B[None,:]

model.write_ic_ds()

## 
model.partition()
model.sun_verbose_flag='-v'
model.run_simulation()

