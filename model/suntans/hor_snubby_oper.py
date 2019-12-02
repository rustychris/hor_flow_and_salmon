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
os.path.exists(cache_dir) or os.makedirs(cache_dir)


def base_model(run_dir,run_start,run_stop,
               restart_from=None,
               steady=False):
    if restart_from is None:
        model=drv.SuntansModel()
        model.load_template("sun-template.dat")
        model.num_procs=num_procs
    else:
        old_model=drv.SuntansModel.load(restart_from)
        model=old_model.create_restart()

    model.manual_z_offset=-4
    model.z_offset=0.0 # moving away from the semi-automated datum shift to manual shift
    
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
        dt=0.25 # for lower friction run on refined shore grid
        model.config['dt']=dt

        model.config['nonlinear']=1

        if 0:
            model.config['Nkmax']=1
            model.config['stairstep']=0 
        if 1:
            model.config['Nkmax']=50
            model.config['stairstep']=1 # 3D, stairstep
        
        model.config['thetaM']=-1
        model.config['z0B']=0.001
        model.config['nu_H']=0.0
        model.config['nu']=1e-5
        model.config['turbmodel']=1 # 1: my25, 10: parabolic
        model.config['CdW']=0.0
        model.config['wetdry']=1
        model.config['maxFaces']=4
        model.config['mergeArrays']=1 # hopefully easier to use than split up.

        # 2019-07-11: refine near-shore swaths of grid, snubby-01 => snubby-04
        # 2019-07-12: further refinements 04 => 06
        grid_src="../grid/snubby_junction/snubby-06.nc"
        grid_bathy="snubby-with_bathy.nc"

        if utils.is_stale(grid_bathy,[grid_src]):
            g_src=unstructured_grid.UnstructuredGrid.from_ugrid(grid_src)
            add_bathy.add_bathy(g_src)
            grid_roughness.add_roughness(g_src)
            g_src.write_ugrid(grid_bathy,overwrite=True)
            
        g=unstructured_grid.UnstructuredGrid.from_ugrid(grid_bathy)
        g.orient_edges()
        g.cells['z_bed'] += model.manual_z_offset
        g.edges['edge_z_bed'] += model.manual_z_offset
            
        model.set_grid(g)
        model.grid.modify_max_sides(4)
    dt=float(model.config['dt'])

    model.config['ntout']=int(1800./dt)
    model.config['ntoutStore']=int(3600./dt)
    # isn't this just duplicating the setting from above?
    model.z_offset=0.0 # moving away from the semi-automated datum shift to manual shift

    model.add_gazetteer("../grid/snubby_junction/forcing-snubby-01.shp")

    def fill(da):
        return utils.fill_tidal_data(da,fill_time=False)

    # TODO: switch to water library for better QA

    if steady:
        # Would like to ramp these up at the very beginning.
        # Mossdale, tested at 220.0 m3/s
        Q_upstream=drv.FlowBC(name='SJ_upstream',Q=220.0,dredge_depth=None)
        Q_downstream=drv.FlowBC(name='SJ_downstream',Q=-100,dredge_depth=None)
        # had been 1.75.  but that's showing about 0.75m too much water.
        # could be too much friction, or too high BC.
        h_old_river=drv.StageBC(name='Old_River',z=1.75 + model.manual_z_offset)
    else:
        Q_upstream=drv.CdecFlowBC(name="SJ_upstream",station="MSD",dredge_depth=None,
                                  filters=[dfm.Transform(fn_da=fill)],
                                  cache_dir=cache_dir)
        # flip sign to get outflow.
        Q_downstream=drv.CdecFlowBC(name="SJ_downstream",station="SJD",dredge_depth=None,
                                    filters=[dfm.Transform(fn=lambda x: -x,fn_da=fill)],
                                    cache_dir=cache_dir)
        h_old_river=drv.CdecStageBC(name='Old_River',station="OH1",cache_dir=cache_dir,
                                    filters=[dfm.Transform(fn=lambda x: x+model.manual_z_offset)])

    model.add_bcs([Q_upstream,Q_downstream,h_old_river])

    model.write()

    assert np.all(np.isfinite(model.bc_ds.boundary_Q.values))

    if not model.restart:
        # bathy rms ranges from 0.015 to 1.5
        # 0.5 appears a little better than 1.0 or 0.1
        # cfg007 used 0.1, and the shape was notably not as good
        # as steady008.
        # cfg008 will return to 0.5...
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
    import argparse
    parser = argparse.ArgumentParser(description='Set up and run Head of Old River simulations.')
    parser.add_argument("-s", "--start", help="Date of simulation start",
                        default="2018-03-10")
    parser.add_argument("-e", "--end", help="Date of simulation stop",
                        default="2018-04-12")
    parser.add_argument("-d", "--dir", help="Run directory",
                        default=None)
    parser.add_argument("-r", "--resume", help="Resume from run",
                        default=None)
    parser.add_argument("-p", "--prefix", help="Prefix for naming set of runs",
                        default="cfg006")
    parser.add_argument("-n", "--dryrun", help="Do not actually partition or run the simulation",
                        action='store_true')
    parser.add_argument("--steady",help="Use constant flows",
                        action='store_true')
    parser.add_argument("--interval",help="Interval for multiple shorter runs, e.g. 1D for 1 day",
                        default=None)

    args=parser.parse_args()
    
    # range of tag data is
    # 2018-03-16 23:53:00 to 2018-04-11 14:11:00
    # for ptm output, start a bit earlier
    multi_run_start=np.datetime64(args.start)
    multi_run_stop=np.datetime64(args.end)

    run_start=multi_run_start
    print(run_start)

    # series of 1-day runs
    # run_interval=np.timedelta64(1,'D')
    # But restarts are not good with average output.
    # Kludge it and run the whole period in one go.
    if args.interval is not None:
        # annoying that this can only be integer values.
        # possible inputs:
        # 5D for 5 days
        # 12h for 12 hours
        # 1800s for half an hour
        run_interval=np.timedelta64(int(args.interval[:-1]),args.interval[-1])
    else:
        # in one go.
        run_interval=multi_run_stop-run_start

    if args.dir is None:
        args.dir="runs/" + args.prefix
        
    run_count=0
    last_run_dir=None
    while run_start < multi_run_stop:
        run_stop=run_start+run_interval
        print(run_start,run_stop)
        date_str=utils.to_datetime(run_start).strftime('%Y%m%d')
        # cfg000: first go
        # cfg001: 2D
        # cfg002: 3D
        # cfg003: fix grid topology in suntans, and timesteps
        # cfg004: run in one go, rather than daily restarts.
        
        run_dir=f"{args.dir}_{date_str}"

        if not drv.SuntansModel.run_completed(run_dir):
            model=base_model(run_dir,run_start,run_stop,
                             restart_from=last_run_dir,
                             steady=args.steady)
            model.sun_verbose_flag="-v"
            try:
                script=__file__
            except NameError:
                script=None
            if script:
                shutil.copyfile(script,os.path.join(model.run_dir,script))
            else:
                print("Could not copy script")
            if args.dryrun:
                print("Dry run - dropping out of loop")
                break
            else:
                model.run_simulation()
        run_count+=1
        last_run_dir=run_dir
        run_start=run_stop


        
