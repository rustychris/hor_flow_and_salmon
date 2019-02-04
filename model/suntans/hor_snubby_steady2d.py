"""
Driver script for Suntans Head of Old River runs.

This version uses the snubby grid, and is a steady, 2D
run used for interpolation
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
from stompy.spatial import wkb2shp, field
from stompy.grid import unstructured_grid

from stompy.model.suntans import sun_driver as drv
##

import stompy.model.delft.dflow_model as dfm

model=drv.SuntansModel()
model.projection="EPSG:26910"

model.num_procs=4
model.sun_bin_dir="/home/rusty/src/suntans/main"
model.mpi_bin_dir="/usr/bin"

model.load_template("sun-template.dat")

# 00: initial
# 01: no friction
# 02: set ntaverage same as ntout.  Unclear on usage.
# 03: fixed averaging bugs in source.
model.set_run_dir(os.path.join('runs',"snubby_steady2d_03"),
                  mode='askclobber')

model.run_start=np.datetime64('2017-03-01 00:00:00')
model.run_stop=np.datetime64('2017-03-01 04:00:00')
ramp_hours=0.5 # how quickly to increase the outflow

model.config['Nkmax']=1

dt=0.5
model.config['dt']=dt
model.config['ntout']=int(900./dt)
# does this make a difference?
model.config['ntaverage']=int(900./dt)
model.config['ntoutStore']=int(3600./dt)

model.config['nonlinear']=0
model.config['stairstep']=0 # 2D, partial step
model.config['thetaM']=-1
model.config['z0B']=0.0
model.config['CdB']=0.0
model.config['CdW']=0.0
model.config['nu_H']=0.0
model.config['nu']=1e-5
model.config['turbmodel']=10 # 1: my25, 10: parabolic
model.config['wetdry']=1

grid_src="../grid/snubby_junction/snubby-01.nc"
grid_bathy="snubby-with_bathy.nc"

import add_bathy
#import grid_roughness

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

# # bathy rms ranges from 0.015 to 1.5
# cell_z0B=0.5*model.grid.cells['bathy_rms']
# e2c=model.grid.edge_to_cells()
# nc1=e2c[:,0]
# nc2=e2c[:,1]
# nc2[nc2<0]=nc1[nc2<0]
# edge_z0B=0.5*( cell_z0B[nc1] + cell_z0B[nc2] )
# model.ic_ds['z0B']=('time','Ne'), edge_z0B[None,:]
# model.write_ic_ds()

## 
model.partition()
model.sun_verbose_flag='-v'
model.run_simulation()

