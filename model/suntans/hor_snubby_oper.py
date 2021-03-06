"""
Driver script for Suntans Head of Old River runs.

This version uses the snubby grid, and includes logic for running
variable flows
 2020-10-08: option to use larger grid
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
import common

from stompy.model.suntans import sun_driver as drv
import stompy.model.hydro_model as hm
import stompy.model.delft.dflow_model as dfm
import add_bathy
import grid_roughness

##

here=os.path.dirname(__file__)

with open(os.path.join(here,'local_config.py')) as fp:
    exec(fp.read())

##
os.path.exists(cache_dir) or os.makedirs(cache_dir)
from stompy.model import slurm_mixin
class SuntansModel(drv.SuntansModel,slurm_mixin.SlurmMixin):
    sun_bin_dir=os.path.join(os.environ['HOME'],"src/suntans/main")

def base_model(run_dir,run_start,run_stop,
               restart_from=None,
               grid_option=None,
               steady=False):
    if restart_from is None:
        model=SuntansModel()
        model.load_template(os.path.join(here,"sun-template.dat"))
        model.num_procs=num_procs
    else:
        old_model=SuntansModel.load(restart_from)
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

    model.config['grid_option']=grid_option or 'snubby'

    if restart_from is None:
        ramp_hours=1 # how quickly to increase the outflow
    else:
        ramp_hours=0

    if not model.restart:
        # dt=0.25 # for lower friction run on refined shore grid
        # with new grid that's deformed around the barrier can probably
        # get away with 0.5
        # dt=0.5 # for lower friction run on refined shore grid
        dt=0.25 # bit of safety for high res grid.
        # dt=0.1 # for shaved cell grid. still not stable.
        model.config['dt']=dt
        model.config['metmodel']=0 # 0: no wind, 4: wind only

        model.config['nonlinear']=1

        if 0:
            model.config['Nkmax']=1
            model.config['stairstep']=0 
        if 1:
            model.config['Nkmax']=50
            model.config['stairstep']=1 # 3D, stairstep
        
        model.config['thetaM']=-1 # with 1, seems to be unstable
        model.config['z0B']=5e-4 # maybe with ADCP bathy need more friction
        # slow, doesn't make a huge difference, but does make w nicer.
        model.config['nonhydrostatic']=0
        model.config['nu_H']=0.0
        model.config['nu']=1e-5
        model.config['turbmodel']=10 # 1: my25, 10: parabolic
        model.config['CdW']=0.0 # 
        model.config['wetdry']=1
        model.config['maxFaces']=4
        model.config['mergeArrays']=1 # hopefully easier to use than split up.

        if model.config['grid_option']=='snubby':
            # 2019-07-11: refine near-shore swaths of grid, snubby-01 => snubby-04
            # 2019-07-12: further refinements 04 => 06
            #grid_src="../grid/snubby_junction/snubby-06.nc"
            #grid_src="../grid/snubby_junction/snubby-07-edit45.nc"
            #grid_src="../grid/snubby_junction/snubby-08-edit06.nc" 
            #grid_src="../grid/snubby_junction/snubby-08-edit24.nc"
            #grid_src="../grid/snubby_junction/snubby-08-edit50.nc" # good
            grid_src="../grid/snubby_junction/snubby-08-edit60.nc" # higher res in hole
            #grid_src="../grid/snubby_junction/snubby-08-refine-edit03.nc" # double res of edit60
        elif model.config['grid_option']=='large':
            grid_src="../grid/merge_v16/merge_v16_edit09.nc"
        else:
            raise Exception("Unknown grid option %s"%grid_option)

        # bathy_suffix=''
        # post_suffix='med0' # on-grid bathy postprocessesing:

        #bathy_suffix='smooth' # pre-smoothed just in sand wave areas
        bathy_suffix='smoothadcp' # pre-smoothed just in sand wave areas, and revisit ADCP data.
        #bathy_suffix='smoothadcp2' # same, but extend ADCP data upstream.
        post_suffix=''
        
        # bathy_suffix='adcp'
        # post_suffix=''

        grid_bathy=os.path.basename(grid_src).replace('.nc',f"-with_bathy{bathy_suffix}{post_suffix}.nc")

        if utils.is_stale(grid_bathy,[grid_src]):
            g_src=unstructured_grid.UnstructuredGrid.from_ugrid(grid_src)
            g_src.cells_center(refresh=True)
            bare_edges=g_src.orient_edges(on_bare_edge='return')
            if len(bare_edges):
                ec=g_src.edges_center()
                for j in bare_edges[:50]:
                    print("Bare edge: j=%d  xy=%g %g"%(j, ec[j,0], ec[j,1]))
                raise Exception('Bare edges in grid')
            add_bathy.add_bathy(g_src,suffix=bathy_suffix)
            add_bathy.postprocess(g_src,suffix=post_suffix)
            
            g_src.write_ugrid(grid_bathy,overwrite=True)
            
        g=unstructured_grid.UnstructuredGrid.from_ugrid(grid_bathy)
        g.cells['z_bed'] += model.manual_z_offset

        g.delete_edge_field('edge_z_bed')
        #g.edges['edge_z_bed'] += model.manual_z_offset

        model.set_grid(g)
        model.grid.modify_max_sides(4)
    dt=float(model.config['dt'])

    # with 0.01, was getting many CmaxW problems.
    model.config['dzmin_surface']=0.05 # may have to bump this up..
    model.config['ntout']=int(1800./dt) 
    model.config['ntaverage']=int(1800./dt)
    model.config['ntoutStore']=int(86400./dt)
    # isn't this just duplicating the setting from above?
    model.z_offset=0.0 # moving away from the semi-automated datum shift to manual shift

    if model.config['grid_option']=='snubby':
        model.add_gazetteer(os.path.join(here,"../grid/snubby_junction/forcing-snubby-01.shp"))
    elif model.config['grid_option']=='large':
        model.add_gazetteer(os.path.join(here,"../grid/merge_v16/forcing-merge_16-01.shp"))
        
    def fill(da):
        return utils.fill_tidal_data(da,fill_time=False)

    if steady:
        # Would like to ramp these up at the very beginning.
        # Mossdale, tested at 220.0 m3/s
        Q_upstream=drv.FlowBC(name='SJ_upstream',flow=220.0,dredge_depth=None)
        Q_downstream=drv.FlowBC(name='SJ_downstream',flow=-100,dredge_depth=None)
        # had been 1.75.  but that's showing about 0.75m too much water.
        # could be too much friction, or too high BC.
        h_old_river=drv.StageBC(name='Old_River',water_level=1.75 + model.manual_z_offset)
    else:
        # 15 minute data. The flows in particular have enough overtide energy
        # that the lowpass interval shouldn't be too large. Spot checks of plots
        # show 1.0h is pretty good
        lp_hours=1.0
        
        data_upstream=common.msd_flow(model.run_start,model.run_stop)
        Q_upstream=drv.FlowBC(name="SJ_upstream",flow=data_upstream.flow_m3s,dredge_depth=None,
                              filters=[hm.Lowpass(cutoff_hours=1.0)])

        data_downstream=common.sjd_flow(model.run_start,model.run_stop)
        # flip sign to get outflow.
        Q_downstream=drv.FlowBC(name="SJ_downstream",flow=-data_downstream.flow_m3s,dredge_depth=None,
                                filters=[hm.Lowpass(cutoff_hours=1.0)])

        or_stage=common.oh1_stage(model.run_start,model.run_stop)
        h_old_river=drv.StageBC(name='Old_River',water_level=or_stage.stage_m,
                                filters=[hm.Transform(fn=lambda x: x+model.manual_z_offset),
                                         hm.Lowpass(cutoff_hours=1.0)])

    model.add_bcs([Q_upstream,Q_downstream,h_old_river])
    
    model.write()

    assert np.all(np.isfinite(model.bc_ds.boundary_Q.values))

    if not model.restart:
        # bathy rms ranges from 0.015 to 1.5
        # 0.5 appears a little better than 1.0 or 0.1
        # cfg007 used 0.1, and the shape was notably not as good
        # as steady008.
        # cfg008 will return to 0.5...
        # BUT - now that I have more metrics in place, it looks like cfg007
        # was actually better, in terms of MAE over top 2m, and smaller bias.
        # with that in mind, steady012 will try an even smaller factor, and infact
        # something more in line with Nikuradse.
        # best roughness using constant z0B was 5e-4.
        # so map 0.1 to that...
        if 1:
            print("Adding roughness")
            grid_roughness.add_roughness(model.grid)
            cell_z0B=(1./30)*model.grid.cells['bathy_rms']
            e2c=model.grid.edge_to_cells()
            nc1=e2c[:,0]
            nc2=e2c[:,1]
            nc2[nc2<0]=nc1[nc2<0]
            edge_z0B=0.5*( cell_z0B[nc1] + cell_z0B[nc2] )
            model.ic_ds['z0B']=('time','Ne'), edge_z0B[None,:]
        model.write_ic_ds()

        if int(model.config['metmodel'])>=4:
            # and how about some wind?
            times=np.array( [model.run_start - np.timedelta64(10,'D'),
                             model.run_start,
                             model.run_stop,
                             model.run_stop  + np.timedelta64(10,'D')] )
            nt=len(times)
            met_ds=model.zero_met(times=times)
            pnts=model.grid.cells_center()[:1]
            for comp in ['Uwind','Vwind']:
                met_ds['x_'+comp]=("N"+comp,),pnts[:,0]
                met_ds['y_'+comp]=("N"+comp,),pnts[:,1]
                met_ds['z_'+comp]=("N"+comp,),10*np.ones_like(pnts[:,1])

            met_ds['Uwind']=('nt',"NUwind"), 0.0 * np.ones((nt,len(pnts)))
            met_ds['Vwind']=('nt',"NVwind"),-5.0 * np.ones((nt,len(pnts)))
            model.met_ds=met_ds
            model.write_met_ds()

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
                        default=None,required=True)
    parser.add_argument("-r", "--resume", help="Resume from run",
                        default=None)
    parser.add_argument("-n", "--dryrun", help="Do not actually partition or run the simulation",
                        action='store_true')
    parser.add_argument("--steady",help="Use constant flows",
                        action='store_true')
    parser.add_argument("-i","--interval",help="Interval for multiple shorter runs, e.g. 1D for 1 day",
                        default=None)
    parser.add_argument("-l","--large",help="Select larger domain",action='store_true')
    
    # args="-l -d runs/largetest00 -s 2018-04-05T12:00 -e 2018-04-05T16:00 ".split()
    args=None
    args=parser.parse_args(args=args)
    
    # range of tag data is
    # 2018-03-16 23:53:00 to 2018-04-11 14:11:00
    # for ptm output, start a bit earlier
    if args.resume is not None:
        # Go ahead and get the restart time, so that intervals and directory names
        # can be chosen below
        last_run_dir=args.resume
        last_model=drv.SuntansModel.load(last_run_dir)
        multi_run_start=last_model.restartable_time()
        print("Will resume run in %s from %s"%(last_run_dir,multi_run_start))
    else:
        multi_run_start=np.datetime64(args.start)
        print("Run start: ",multi_run_start)
        last_run_dir=None
        
    run_start=multi_run_start
    multi_run_stop=np.datetime64(args.end)

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
        run_interval=None

    assert args.dir is not None
        
    run_count=0
    # For restarts, we don't yet know the run_start
    while True:
        if last_run_dir is not None:
            last_run=drv.SuntansModel.load(last_run_dir)
            # Have to be careful that run is long enough to output new
            # restart time step, otherwise we'll get stuck.
            run_start=last_run.restartable_time()
        if run_start>=multi_run_stop:
            break

        if run_interval is not None:
            run_stop=min(run_start+run_interval,multi_run_stop)
        else:
            run_stop=multi_run_stop
        print("Simulation period: %s -- %s"%(run_start,run_stop))
        date_str=utils.to_datetime(run_start).strftime('%Y%m%d')
        # cfg000: first go
        # cfg001: 2D
        # cfg002: 3D
        # cfg003: fix grid topology in suntans, and timesteps
        # cfg004: run in one go, rather than daily restarts.
        # cfg009: flows from WDL, very little friction, nonhydrostatic.
        #         more reasonable dzmin_surface.
        # cfg010: new grid (8.60), many tweaks, new bathy.
        # cfg011: lowpass boundary conditions
        
        run_dir=f"{args.dir}_{date_str}"

        if not drv.SuntansModel.run_completed(run_dir):
            grid_option='snubby'
            if args.large:
                grid_option='large'
            model=base_model(run_dir,run_start,run_stop,
                             restart_from=last_run_dir,
                             grid_option=grid_option,
                             steady=args.steady)
            model.sun_verbose_flag="-v"
            try:
                script=__file__
            except NameError:
                script=None
            if script:
                shutil.copyfile(script,os.path.join(model.run_dir,
                                                    os.path.basename(script)))
            else:
                print("Could not copy script")
            if args.dryrun:
                print("Dry run - dropping out of loop")
                break
            else:
                model.run_simulation()
        run_count+=1
        last_run_dir=run_dir
        if run_interval is None:
            print("Single shot run. Breaking out")
            break



        
