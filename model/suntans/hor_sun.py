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
from stompy.spatial import wkb2shp, field
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
# steady_009: run with updated suntans, lower reference elevation, use z0 instead of CdB.
# steady_010: new quad/tri grid with mixed resolution.
# steady_011: updated bathy, and see if advection redistributes velocities
# steady_012: short run with half time step to diagnose stripes.
# steady_013: back to 4h run, kill stripes with thetaM=-1
# steady_014: 5x increase in z0B to 0.005
# steady_015: 5x decrease in z0B to 0.0002
# steady_016: decrease z0B to 0.00005, nu_H=0.0
#       barely unstable.
# steady_017: z0B back to 0.001, nu_H=0.0, but in flows up to 250
# steady_018: enable avg output.  test with nonlinear=0.
# steady_019: stairstep=1, still nonlinear=0
# steady_020: stairstep=1, return to nonlinear=1
# steady_021: stairstep=1, nonlinear=1, now add some Kh
# steady_022: stairstep=0, nonlinear=1, Nk=1, Kh=0
# steady_023: change vertical advection of hor-mom at surface
# steady_024: bring in parameters from bend case: nu=0.02, z0B=0.01
#    stairstep=1, back to 3D
#    re-initialized this to use spatially variable z0.
# steady_025: decrease spatial roughness by 10x (0.01 and 0.001)
# steady_026: tighten up the rough polygon, push 0.01 and 0.003 as
#   the values in the shapefile.
# steady_027: lower OldRiver free surface forcing by 0.65m to improve match
# steady_028: increase nu 0.02 => 0.05.  That didn't do much.
# steady_029: drastic nu=0.2: didn't finish.  why?  early output looks like
#   it's not a real improvement, though.
#  Xsteady_030: re-run of original 024: nu=0.02, z0B=0.01
#  X   stairstep=1, back to 3D. no variation in z0B.  not stable!?
# steady_030: retake - friction down to 0.001, also nu down to 0.001,
#     and try fixes in suntans to allow conserveMomentum to coexists with
#     wetdry=1.
# steady_031:  leave z0b at 0.001, but crank up nu to 0.05
#    this makes little difference.  a very slight evening out of the
#    profile, <10% what is needed.
# steady_032:  increase z0b to 0.005 (0.01 is hard to make stable downstream)
#    - have to ramp up the outflow for this to work.  even then it's not stable.
#    - this is not too bad...
# steady_033: back down 030 values, but try out CdW=0.1
# steady_034: CdW was good, try some more CdW=0.2 (CdW=0.3 not stable)
# steady_035: back to 030 values, and now suntans does not advect turbulence quantities
#   that was super noisy
# steady_036: try parabolic eddy viscosity -- dropping nu down to only 10x molecular
#  keep z0B=0.001
# steady_037: parabolic, but bump z0B up to 0.01

# Suite of runs to reassess status:
# all using parabolic.
# steady_038: 2D, z0B=0.001
# steady_039: 3D, z0B=0.001, with advection.
# steady_040: 3D, z0B=0.001, no advection
# steady_041: 2D no advection.
model.set_run_dir(os.path.join('runs',"steady_041"),
                  mode='askclobber')

model.run_start=np.datetime64('2017-03-01 00:00:00')
model.run_stop=np.datetime64('2017-03-01 04:00:00')
ramp_hours=5 # how quickly to increase the outflow

model.config['nonlinear']=0
model.config['Nkmax']=1
if int(model.config['Nkmax'])==1:
    model.config['stairstep']=0 # 2D, partial step
else:
    model.config['stairstep']=1 # 3D, stairstep
model.config['thetaM']=-1
model.config['z0B']=0.001
model.config['nu_H']=0.0
model.config['nu']=1e-5
model.config['turbmodel']=10
model.config['CdW']=0.0
model.config['wetdry']=1

dt=0.5 # for low-friction case 1.0 was too long
model.config['dt']=dt
model.config['ntout']=int(1800./dt)
model.config['ntoutStore']=int(3600./dt)

grid_src="../grid/full_merge_v15/edit10.nc"
grid_bathy="grid-with_bathy.nc"
import add_bathy
if not os.path.exists(grid_bathy) or os.stat(grid_bathy).st_mtime < os.stat(grid_src).st_mtime:
    g_src=unstructured_grid.UnstructuredGrid.from_ugrid(grid_src)
    add_bathy.add_bathy(g_src)
    g_src.write_ugrid(grid_bathy,overwrite=True)

model.set_grid(unstructured_grid.UnstructuredGrid.from_ugrid(grid_bathy))

model.grid.modify_max_sides(4)

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

model.add_gazetteer("gis/forcing-v00.shp")

Q_upstream=drv.FlowBC(name='SJ_upstream',Q=250.0)
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
Q_downstream=drv.FlowBC(name='SJ_downstream',Q=Qdown)
h_old_river=drv.StageBC(name='Old_River',z=2.2-0.65)

model.add_bcs([Q_upstream,Q_downstream,h_old_river])

model.write()

#-- 
if 0:
    # Add in variable roughness
    ec=model.grid.edges_center()
    z0_map=field.GdalGrid("../../bathy/composite-roughness.tif")

    z0=z0_map(ec)
    z0[np.isnan(z0)]=float(model.config['z0B'])
    model.ic_ds['z0B']=('time','Ne'),z0[None,:]
    model.write_ic_ds()

#

model.partition()
model.sun_verbose_flag='-v'
model.run_simulation()


# with the lower friction, get a vertical courant number of 5.25
# due to a cell that's 1mm thick.
