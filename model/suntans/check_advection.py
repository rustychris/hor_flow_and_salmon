"""
Driver script for Suntans Head of Old River runs -- specific
to advection tests
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

old_model=drv.SuntansModel.load('runs/steady_011/suntans.dat')
model=old_model.create_restart()

model.projection="EPSG:26910"

model.sun_bin_dir="/home/rusty/src/suntans/main"
model.mpi_bin_dir="/usr/bin"

if 0:
    model.set_run_dir(os.path.join('runs',"steady_011A"),
                      mode='pristine')
    log.warning("Disabling advection for steady_011")
    model.config['nonlinear']=0

if 0:
    model.set_run_dir(os.path.join('runs',"steady_011B"),
                      mode='pristine')
    model.config['nonlinear']=1

if 0:
    # this decrease velocity everywhere, but does not
    # remove the left-hugging jet.
    # maybe depth-dependence of friction is not right?
    model.set_run_dir(os.path.join('runs',"steady_011C"),
                      mode='pristine')
    model.config['nonlinear']=1
    model.config['z0B']=0.01 # up from 0.001

if 0:
    # this crashed pretty quickly
    model.set_run_dir(os.path.join('runs',"steady_011D"),
                      mode='pristine')
    model.config['nonlinear']=1
    model.config['z0B']=0.001
    model.config['nu_H']=1.0 # up from 0.1

if 0:
    # this completes
    model.set_run_dir(os.path.join('runs',"steady_011E"),
                      mode='pristine')
    model.config['nonlinear']=1
    model.config['z0B']=0.001
    model.config['nu_H']=0.3 # split the difference

if 0:
    model.set_run_dir(os.path.join('runs',"steady_011F"),
                      mode='pristine')
    model.config['nonlinear']=1
    model.config['conserveMomentum']=0

if 0:
    model.set_run_dir(os.path.join('runs',"steady_011G"),
                      mode='pristine')
    model.config['nonlinear']=1
    model.config['thetaM']=0.6

if 1:
    model.set_run_dir(os.path.join('runs',"steady_011H"),
                      mode='pristine')
    model.config['nonlinear']=1
    model.config['thetaM']=-1

model.run_start=old_model.restartable_time()
model.run_stop=model.run_start+np.timedelta64(900,'s')

dt=float(model.config['dt'])
model.config['ntout']=int(900./dt)
##
# Annoying, but suntans doesn't like signed elevations
# this offset will be applied to grid depths and freesurface boundary conditions.
# this was -10, but that leaves a lot of dead space on top.
model.z_offset=-4

model.add_gazetteer("gis/forcing-v00.shp")

Q_upstream=drv.FlowBC(name='SJ_upstream',Q=210.0)
Q_downstream=drv.FlowBC(name='SJ_downstream',Q=-100.0)
h_old_river=drv.StageBC(name='Old_River',z=2.2)


model.add_bcs([Q_upstream,Q_downstream,h_old_river])

model.write()
model.partition()
model.run_simulation()

##
from stompy.spatial import wkb2shp
from stompy import xr_transect

tran_shp="../../gis/model_transects.shp"
tran_geoms=wkb2shp.shp2geom(tran_shp)

##
run_dir=model.run_dir

# plot a few a sections
six.moves.reload_module(drv)
run=drv.SuntansModel.load(run_dir)

##
utils.path("../../field/adcp")
import summarize_xr_transects
six.moves.reload_module(summarize_xr_transects)


##
fig_dir=os.path.join(model.run_dir,'figs-20180927')
os.path.exists(fig_dir) or os.makedirs(fig_dir)

for t in tran_geoms[3:4]:
    print(t['name'])
    xy=np.array(t['geom'])
    tran=run.extract_transect(xy=xy,time=0,dx=2)
    wet_samples=np.nonzero(np.abs(tran.z_dz.sum(dim='layer').values)>0.01)[0]
    sample_slice=slice(wet_samples[0],wet_samples[-1]+1)
    tran=tran.isel(sample=sample_slice)
    fig=summarize_xr_transects.summarize_transect(tran,num=2,w_scale=0.0)
    #fig.canvas.draw()
    #plt.pause(0.1)
    #fig.savefig(os.path.join(fig_dir,"transect-%s.png"%t['name']))


##
