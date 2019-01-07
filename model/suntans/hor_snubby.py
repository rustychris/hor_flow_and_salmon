"""
Driver script for Suntans Head of Old River runs.

This version uses the snubby grid
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
#        061: 
model.set_run_dir(os.path.join('runs',"snubby_060"),
                  mode='askclobber')

model.run_start=np.datetime64('2017-03-01 00:00:00')
model.run_stop=np.datetime64('2017-03-01 04:00:00')
ramp_hours=1 # how quickly to increase the outflow
double_grid=0 # how many times to double the grid.
shuffle_edges=0
smooth_bathy=0

model.config['Nkmax']=50

if not double_grid:
    # for low-friction case 1.0 was too long
    # for high friction, 3D case, 0.5 was too long
    # actually, it crashes regardless.  so crash faster, with
    # 0.5
    dt=0.5
else:
    dt=0.25
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
if int(model.config['nonlinear'])==2:
    # Oliver suggests u^2 dt/2
    U=0.5 # 1.0 was stable.
    model.config['nu_H']=U**2 * dt/2.0
else:
    model.config['nu_H']=0.0
model.config['nu']=1e-5
model.config['turbmodel']=10
model.config['CdW']=0.0
model.config['wetdry']=1


if not double_grid: # original
    grid_src="../grid/snubby_junction/snubby-01.nc"
    grid_bathy="snubby-with_bathy.nc"
else: # doubled
    grid_src="snubby-doubled-edit04.nc"
    grid_bathy="snubby-doubled-with_bathy.nc"

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
if shuffle_edges:
    print("Shuffling edge orientation")
    g.edge_to_cells()
    for j in range(g.Nedges()):
        if g.edges['cells'][j,1]<0: continue # boundary
        if np.random.random() < 0.5:
            rec=g.edges[j]
            g.modify_edge( j=j,
                           nodes=[rec['nodes'][1],rec['nodes'][0]],
                           cells=[rec['cells'][1],rec['cells'][0]] )

if smooth_bathy:
    d=g.cells['cell_depth']
    M=g.smooth_matrix()
    for i in range(smooth_bathy):
        print("Smooth!")
        d=M.dot(d)
    # g.cells['cell_depth'][:]=d
    g.add_cell_field('depth',d,on_exists='overwrite')
    g.delete_node_field('depth')

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

Q_upstream=drv.FlowBC(name='SJ_upstream',Q=220.0)
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
# 2.5 crashed.
# h_old_river=drv.StageBC(name='Old_River',z=2.2-0.65)
h_old_river=drv.StageBC(name='Old_River',z=2.5)

model.add_bcs([Q_upstream,Q_downstream,h_old_river])


## 
model.write()

# bathy rms ranges from 0.015 to 1.5
cell_z0B=1.0*model.grid.cells['bathy_rms']
e2c=model.grid.edge_to_cells()
nc1=e2c[:,0]
nc2=e2c[:,1]
nc2[nc2<0]=nc1[nc2<0]
edge_z0B=0.5*( cell_z0B[nc1] + cell_z0B[nc2] )
model.ic_ds['z0B']=('time','Ne'), edge_z0B[None,:]

model.write_ic_ds()

## 
model.partition()
model.sun_verbose_flag='-vv'
model.run_simulation()


