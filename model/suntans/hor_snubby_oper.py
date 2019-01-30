"""
Driver script for Suntans Head of Old River runs.

This version uses the snubby grid, and includes logic for running
variable flows
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
from stompy.grid import unstructured_grid

from stompy.model.suntans import sun_driver as drv
import stompy.model.delft.dflow_model as dfm
import add_bathy
import grid_roughness

##

six.moves.reload_module(utils)
six.moves.reload_module(dfm)
six.moves.reload_module(unstructured_grid)
six.moves.reload_module(drv)

with open('local_config.py') as fp:
    exec(fp.read())

##


def base_model(run_dir,run_start,run_stop,
               restart_from=None):
    if restart_from is None:
        model=drv.SuntansModel()
        model.load_template("sun-template.dat")
        model.num_procs=4
    else:
        old_model=drv.SuntansModel.load(restart_from)
        model=old_model.create_restart()
        
    model.projection="EPSG:26910"

    model.set_run_dir(run_dir,mode='pristine')

    if model.restart:
        model.run_start=model.restart_model.restartable_time()
    else:
        model.run_start=run_start
    
    model.run_stop=run_stop

    if restart_from is None:
        ramp_hours=1 # how quickly to increase the outflow
    else:
        ramp_hours=0

    if not model.restart:
        dt=0.5
        model.config['dt']=dt

        model.config['nonlinear']=1

        if 1:
            model.config['Nkmax']=1
            model.config['stairstep']=0 
        if 0:
            model.config['Nkmax']=50
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

        if not os.path.exists(grid_bathy) or os.stat(grid_bathy).st_mtime < os.stat(grid_src).st_mtime:
            g_src=unstructured_grid.UnstructuredGrid.from_ugrid(grid_src)
            add_bathy.add_bathy(g_src)
            grid_roughness.add_roughness(g_src)
            g_src.write_ugrid(grid_bathy,overwrite=True)
            
        g=unstructured_grid.UnstructuredGrid.from_ugrid(grid_bathy)
        g.orient_edges()
        model.set_grid(g)
        model.grid.modify_max_sides(4)
    dt=float(model.config['dt'])

    model.config['ntout']=int(1800./dt)
    model.config['ntoutStore']=int(3600./dt)
    model.z_offset=-4

    model.add_gazetteer("../grid/snubby_junction/forcing-snubby-01.shp")

    # HERE -- these need some data filling.
    # mossdale flow has some nan, which should be filled.
    # something like:
    #   da_fill=utils.fill_tidal_data( ds.boundary_Q.isel(Nseg=0),fill_time=False )
    # should get most of the way there

    def fill(da):
        return utils.fill_tidal_data(da,fill_time=False)
    
    # Would like to ramp these up at the very beginning.
    # Mossdale, tested at 220.0 m3/s
    Q_upstream=drv.CdecFlowBC(name="SJ_upstream",station="MSD",dredge_depth=None,
                              filters=[dfm.Transform(fn_da=fill)],
                              cache_dir=cache_dir)
    # San Joaquin above Dos Reis, tested at -100
    # flip sign to get outflow.

    Q_downstream=drv.CdecFlowBC(name="SJ_downstream",station="SJD",dredge_depth=None,
                                filters=[dfm.Transform(fn=lambda x: -x,fn_da=fill)],
                                cache_dir=cache_dir)
    h=np.timedelta64(1,'h')
    
    h_old_river=drv.CdecStageBC(name='Old_River',station="OH1",cache_dir=cache_dir)

    model.add_bcs([Q_upstream,Q_downstream,h_old_river])

    model.write()

    assert np.all(np.isfinite(model.bc_ds.boundary_Q.values))

    if not model.restart:
        # bathy rms ranges from 0.015 to 1.5
        # 0.5 appears a little better than 1.0 or 0.1
        cell_z0B=0.5*model.grid.cells['bathy_rms']
        e2c=model.grid.edge_to_cells()
        nc1=e2c[:,0]
        nc2=e2c[:,1]
        nc2[nc2<0]=nc1[nc2<0]
        edge_z0B=0.5*( cell_z0B[nc1] + cell_z0B[nc2] )
        model.ic_ds['z0B']=('time','Ne'), edge_z0B[None,:]

        model.write_ic_ds()

    model.partition()
    model.sun_verbose_flag='-v'
    return model

if __name__=='__main__':
    # range of tag data is
    # 2018-03-16 23:53:00 to 2018-04-11 14:11:00
    multi_run_start=np.datetime64("2018-03-16 00:00")
    multi_run_stop=np.datetime64("2018-04-12 00:00")

    run_start=multi_run_start
    print(run_start)

    # series of 1-day runs
    run_count=0
    last_run_dir=None
    while run_start < multi_run_stop:
        run_stop=run_start+np.timedelta64(1,'D')
        print(run_start,run_stop)
        date_str=utils.to_datetime(run_start).strftime('%Y%m%d')
        # cfg000: first go
        # Cfg001: 2D

        run_dir="runs/snubby_cfg001_%s"%date_str

        if not drv.SuntansModel.run_completed(run_dir):
            model=base_model(run_dir,run_start,run_stop,
                             restart_from=last_run_dir)
            model.sun_verbose_flag="-v"
            model.run_simulation()
        run_count+=1
        last_run_dir=run_dir
        run_start+=np.timedelta64(1,'D')



