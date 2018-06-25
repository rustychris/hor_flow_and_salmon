"""
For each of the transects, summarize the data in its raw form. 
"""
from __future__ import print_function

# Read the collection of csv files associated with a sontek transect
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import glob
import six

import read_sontek
# import read_untrim_section
import transect_to_adcpy
from stompy.memoize import memoize
from stompy import utils
from stompy.plot import plot_utils
from stompy.spatial import proj_utils
from stompy import xr_transect, xr_utils

from stompy.model.delft import dfm_grid
import stompy.model.delft.dflow_model as dfm

from adcpy import adcpy
from adcpy import adcpy_recipes

##

six.moves.reload_module(adcpy_recipes)
six.moves.reload_module(adcpy.plot)
six.moves.reload_module(adcpy.util)
six.moves.reload_module(adcpy)
six.moves.reload_module(read_sontek)
six.moves.reload_module(transect_to_adcpy)
six.moves.reload_module(xr_utils)
six.moves.reload_module(xr_transect)
six.moves.reload_module(dfm)

study_zoom=[646000, 649000, 4185000, 4186500]

@memoize()
def bathy():
    # from stompy.spatial import field
    #return field.GdalGrid('../../bathy/OldRvr_at_SanJoaquinRvr2012_0104-utm-m_NAVD.tif')
    utils.path("../../bathy")
    from bathy import dem
    tile=dem().extract_tile(study_zoom)
    return tile

def tran_zoom(ds):
    return utils.expand_xxyy([ds.x_sample.values.min(), ds.x_sample.values.max(),
                              ds.y_sample.values.min(), ds.y_sample.values.max() ],
                             0.1)

def set_bounds(ax,ds):
    ax.axis('equal')
    ax.axis(tran_zoom(ds))


# source='adcp'
# source='untrim'
source='dfm'

fig_dir="figs-20180622-%s"%source
os.path.exists(fig_dir) or os.mkdir(fig_dir)

if source=='adcp':
    # Data from each set of repeats is in a single folder
    transects=glob.glob("040518_BT/*BTref")

    def tweak_sontek(ds):
        """
        """
        if '20180404085311r.rivr' in ds.rivr_filename:
            print("%s is known to be offset along channel"%ds.rivr_filename)
        # that transect is also missing some lat/lon
        bad_sample=(ds.lon.values==0.0)
        if np.any(bad_sample):
            print("%d of %d missing lat/lon"%(bad_sample.sum(),len(bad_sample)))
            ds=ds.isel(sample=~bad_sample)
        return ds

    @memoize(lru=10)
    def read_transect(transect):
        """
        Find and load transects from a folder, calling tweak_sontek on each.
        return list of xr.Dataset
        """
        rivr_fns=glob.glob('%s/*.rivr'%transect) + glob.glob('%s/*.riv'%transect)

        tran_dss=[ tweak_sontek(read_sontek.surveyor_to_xr(fn,proj='EPSG:26910'))
                   for fn in rivr_fns ]
        return tran_dss

    def get_tran_fig_dir(transect):
        return os.path.join(fig_dir,os.path.basename(transect))

    def ds_to_adcpy(ds):
        return read_sontek.ADCPSontekM9(ds=ds)

elif source=='untrim':
    hydro_txt_fn="../../model/untrim/ed-steady/section_hydro.txt"
    names=xr_transect.section_hydro_names(hydro_txt_fn)

    transects=[ (hydro_txt_fn,name) for name in names]

    def read_transect(transect):
        hydro_txt_fn,section_name=transect
        ds=xr_transect.section_hydro_to_transect(hydro_txt_fn,section_name)
        xy=np.c_[ds.x_sample.values,ds.y_sample.values]
        ll=proj_utils.mapper("EPSG:26910","WGS84")(xy)
        ds['lon']=('sample',),ll[:,0]
        ds['lat']=('sample',),ll[:,1]
        return [ds]
    def get_tran_fig_dir(transect):
        return os.path.join(fig_dir,transect[1])
    def ds_to_adcpy(ds):
        return transect_to_adcpy.ADCPXrTransectData(ds=ds)
elif source=='dfm':
    # for now, pull transect xy from the existing untrim sections
    hydro_txt_fn="../../model/untrim/ed-steady/section_hydro.txt"
    names=xr_transect.section_hydro_names(hydro_txt_fn)
    dfm_map=xr.open_dataset(
        '../../'
        'model/dfm/dfm/'
        'runs/hor_002/DFM_OUTPUT_flowfm/flowfm_0000_20120801_000000_map.nc')

    g=dfm_grid.DFMGrid(dfm_map)
    transects=[ (hydro_txt_fn,name,dfm_map,g) for name in names]

    def read_transect(transect):
        hydro_txt_fn,section_name,dfm_map,g=transect
        untrim_ds=xr_transect.section_hydro_to_transect(hydro_txt_fn,section_name)
        line_xy=np.c_[untrim_ds.x_sample.values,untrim_ds.y_sample.values]

        ds=dfm.extract_transect(dfm_map.isel(time=-1),line=line_xy,dx=3,grid=g)

        xy=np.c_[ds.x_sample.values,ds.y_sample.values]

        ll=proj_utils.mapper("EPSG:26910","WGS84")(xy)
        ds['lon']=('sample',),ll[:,0]
        ds['lat']=('sample',),ll[:,1]
        return [ds]
    def get_tran_fig_dir(transect):
        return os.path.join(fig_dir,transect[1])
    def ds_to_adcpy(ds):
        return transect_to_adcpy.ADCPXrTransectData(ds=ds)

dss=read_transect(transects[12])
A=ds_to_adcpy(dss[0])
##

transect_xy_fn=os.path.join(fig_dir,'transects-xy.csv')
os.path.exists(transect_xy_fn) and os.unlink(transect_xy_fn)

for transect in transects:
    tran_fig_dir=get_tran_fig_dir(transect)
    os.path.exists(tran_fig_dir) or os.mkdir(tran_fig_dir)

    tran_dss=read_transect(transect)

    # Plan view quiver for individual repeats:
    zoom_overview=(646804, 647544, 4185572, 4186124)
    ds0=tran_dss[0]
    zoom_tight=tran_zoom(ds0)

    # Choose a coordinate system from the first transect:
    xy=np.c_[ds0.x_sample.values,ds0.y_sample.values]
    xy0=xy[0]
    across_unit=utils.to_unit( xy[-1] - xy[0] )
    # Roughly force to point river right
    mean_u=ds0.Ve.mean()
    mean_v=ds0.Vn.mean()

    # resolve ambiguity to make mean flow positive, roughly assuming
    # ebb/river flow
    if np.dot(across_unit,[mean_v,-mean_u])<0:
        across_unit*=-1
        xy0=xy[-1]
    # Then this is downstream:
    along_unit=np.array( [-across_unit[1],across_unit[0]] )

    def ds_to_linear(ds):
        xy=np.c_[ds.x_sample.values,ds.y_sample.values]
        # pull dimensions from ds, distances from vector product
        return ds.x_sample*0 + np.dot(xy-xy0,across_unit)

    if 1: # Plan view quiver of each individual repeat
        for repeat,ds in enumerate(tran_dss):
            fig=plt.figure(2)
            fig.clf()
            fig.set_size_inches((8,6),forward=True)

            ax=fig.add_axes([0,0,1,1])

            over_ax=fig.add_axes([0,0,0.15,0.15])

            col='g'
            print(repeat)
            avg_east=ds.Ve.mean(dim='layer')
            avg_north=ds.Vn.mean(dim='layer')
            quiv=ax.quiver(ds.x_sample.values, ds.y_sample.values, avg_east.values, avg_north.values,
                           color=col,width=0.001)
            ax.quiverkey(quiv,0.9,0.9,0.5,"0.5 m/s",coordinates='axes')
            ax.text( 0.05, 0.95,ds.source,
                     transform=ax.transAxes,color=col)

            set_bounds(ax,ds)

            for x in ax,over_ax:
                bathy().plot(ax=x,cmap='gray',vmin=-20,vmax=6)
                x.xaxis.set_visible(0)
                x.yaxis.set_visible(0)

            over_ax.axis(zoom_overview)

            over_ax.plot(ds.x_sample.values,ds.y_sample.values,color=col)
            plot_utils.scalebar([0.3,0.03],
                                divisions=[0,10,20,50],label_txt="m",ax=ax,xy_transform=ax.transAxes,dy=0.02)

            fig.savefig(os.path.join(tran_fig_dir,'quiver-2d-repeat%02d.png'%repeat))

    if 0 and len(tran_dss)>1: # Plan view quiver of all repeats together:
        fig=plt.figure(3)
        fig.clf()
        fig.set_size_inches((8,6),forward=True)
        ax=fig.add_axes([0,0,1,1])
        over_ax=fig.add_axes([0,0,0.15,0.15])

        for repeat,ds in enumerate(tran_dss):
            print(repeat)
            if len(tran_dss)>1:
                col=cm.jet( repeat/(len(tran_dss)-1.0))
            else:
                col='b'
            avg_east=ds.Ve.mean(dim='layer')
            avg_north=ds.Vn.mean(dim='layer')
            quiv=ax.quiver(ds.x_sample.values, ds.y_sample.values, avg_east.values, avg_north.values,
                           color=col,width=0.001)
            ax.text( 0.05, 0.95-0.03*repeat,ds.source,
                     transform=ax.transAxes,color=col)

            over_ax.plot(ds.x_sample.values,ds.y_sample.values,color=col)

        for x in ax,over_ax:
            bathy().plot(ax=x,cmap='gray',vmin=-20,vmax=6)
            x.xaxis.set_visible(0)
            x.yaxis.set_visible(0)

        over_ax.axis(zoom_overview)
        ax.quiverkey(quiv,0.9,0.9,0.5,"0.5 m/s",coordinates='axes')
        set_bounds(ax,ds)

        plot_utils.scalebar([0.5,0.03],divisions=[0,10,20,50],label_txt="m",ax=ax,xy_transform=ax.transAxes,dy=0.02)

        fig.savefig(os.path.join(tran_fig_dir,'quiver-2d-allrepeats.png'))

    if 1: # X-Z plot of longitudinal and lateral velocity
        x_lat=ds_to_linear(tran_dss[0])
        if 'depth_bt' in tran_dss[0]:
            z=-tran_dss[0].depth_bt
        else:
            # for untrim output:
            z=-(tran_dss[0].z_surf - tran_dss[0].z_bed)
        zmax=0.0
        zmin=z.min()-0.2
        xmin=x_lat.min() - 3.0
        xmax=x_lat.max() + 3.0

        for repeat,ds in enumerate(tran_dss):
            ds_lateral=ds_to_linear(ds)

            Vlong=ds.Ve*along_unit[0] + ds.Vn*along_unit[1]
            Vlat =ds.Ve*across_unit[0] + ds.Vn*across_unit[1]

            X,Z = xr.broadcast(ds_lateral,ds.z_ctr)

            fig=plt.figure(4)
            fig.clf()
            fig.set_size_inches((10,6),forward=True)
            fig,(ax_lon,ax_lat) = plt.subplots(2,1,num=4,sharex=True,sharey=True)

            scat_lon=ax_lon.scatter( X,Z, 30, Vlong, cmap='jet',label='__nolabel__')
            scat_lon.set_clim([-1,1])
            plt.colorbar(scat_lon,label='Along channel (m/s)',ax=ax_lon)

            scat_lat=ax_lat.scatter( X,Z, 30, Vlat, cmap='jet',label='__nolabel__')
            scat_lat.set_clim([-.5,.5])
            plt.colorbar(scat_lat,label='Across channel (m/s)',ax=ax_lat)

            for ax in [ax_lon,ax_lat]:
                if 'depth_vb' in ds:
                    ax.plot( ds_lateral, -ds.depth_vb,'k-',label='VB depth')
                if 'depth_bt' in ds:
                    ax.plot( ds_lateral, -ds.depth_bt,'k--',label='BT depth')
                ax.legend(fontsize=9)
                ax.set_ylabel('Depth (m)')
            # Just bottom axis
            ax.set_xlabel('Across-channel (m)')

            ax_lon.axis(xmin=xmin,xmax=xmax,ymin=zmin,ymax=zmax)

            # And an overview panel:
            fig.subplots_adjust(left=0.4,right=0.98)

            ax_map=fig.add_axes([0.05,0.2,0.3,0.5])

            bathy().plot(ax=ax_map,cmap='gray',vmin=-20,vmax=6)
            ax_map.xaxis.set_visible(0)
            ax_map.yaxis.set_visible(0)
            set_bounds(ax_map,ds)
            avg_east=ds.Ve.mean(dim='layer')
            avg_north=ds.Vn.mean(dim='layer')
            quiv=ax_map.quiver(ds.x_sample.values, ds.y_sample.values, avg_east.values, avg_north.values,
                               color='g',width=0.001)

            fig.text( 0.03,0.95,ds.attrs['source'])
            fig.savefig(os.path.join(tran_fig_dir,'velocity-yz-repeat%02d.png'%repeat))

    if 0: # X-Z plot of SNR
        x_lat=ds_to_linear(tran_dss[0])
        if 'depth_bt' in tran_dss[0]:
            z=-tran_dss[0].depth_bt
        else:
            # for untrim output:
            z=-(tran_dss[0].z_surf - tran_dss[0].z_bed)
        zmax=0.0
        zmin=z.min()-0.2
        xmin=x_lat.min() - 3.0
        xmax=x_lat.max() + 3.0

        for repeat,ds in enumerate(tran_dss):
            ds_lateral=ds_to_linear(ds)
            snr = ds.snr.mean(dim='beam')
            _,freq,prof_type,cell_size = xr.broadcast(snr,ds.frequency,ds.profile_type,ds.cell_size)

            # Assume SNR is in decibels.  Then twice the distance means really 2x the distance roundtrip,
            # which by inverse square is 4x less signal, just based on spreading.
            # atten_db_per_m = 0.5 # dB/m - very rough from lit.
            # atten_db_per_m = 0.8 # this looks more realistic, though.  maybe from sediment?
            # or even frequency dependent:
            atten_db_per_m=np.where(freq==1,-1.5,1.2)

            dsp_gain_db=7*(np.sqrt(0.2 / cell_size) - 1.0)  + np.where(freq==1,6,0)

            dist0=1.0
            spreading_db=20* ( 2*np.log10( (ds.location/dist0) ) )
            atten_db=(2*ds.location) * atten_db_per_m
            snr_norm=snr + spreading_db + atten_db + dsp_gain_db

            X,Z = xr.broadcast(ds_lateral,ds.location)
            Z=-Z # -ds.location

            fig=plt.figure(8)
            fig.clf()
            fig.set_size_inches((10,6),forward=True)
            fig,(ax_snr,ax_snr_norm) = plt.subplots(2,1,num=8,sharex=True,sharey=True)

            scat_snr=ax_snr.scatter( X,Z, 30, snr, cmap='jet',label='__nolabel__')
            scat_snr2=ax_snr_norm.scatter( X,Z, 30, snr_norm, cmap='jet',label='__nolabel__')

            # scat_snr.set_clim([-1,1])
            plot_utils.cbar(scat_snr,label='SNR',ax=ax_snr)
            plot_utils.cbar(scat_snr2,label='SNR*dist',ax=ax_snr_norm)
            scat_snr2.set_clim([55,65])

            for ax in [ax_snr,ax_snr_norm]:
                #if 'depth_vb' in ds:
                #    ax.plot( ds_lateral, -ds.depth_vb,'k-',label='VB depth')
                if 'depth_bt' in ds:
                    ax.plot( ds_lateral, -ds.depth_bt,'k--',label='BT depth')
                ax.legend(fontsize=9)
                ax.set_ylabel('Depth (m)')
            # Just bottom axis
            ax.set_xlabel('Across-channel (m)')

            ax_snr.axis(xmin=xmin,xmax=xmax,ymin=zmin,ymax=zmax)

            # And an overview panel:
            fig.subplots_adjust(left=0.4,right=0.98,top=0.9,bottom=0.2)

            ax_map=fig.add_axes([0.05,0.2,0.3,0.5])

            bathy().plot(ax=ax_map,cmap='gray',vmin=-20,vmax=6)
            ax_map.xaxis.set_visible(0)
            ax_map.yaxis.set_visible(0)
            set_bounds(ax_map,ds)
            avg_east=ds.Ve.mean(dim='layer')
            avg_north=ds.Vn.mean(dim='layer')
            quiv=ax_map.quiver(ds.x_sample.values, ds.y_sample.values, avg_east.values, avg_north.values,
                               color='g',width=0.001)

            fig.text( 0.03,0.95,ds.attrs['source'])
            fig.savefig(os.path.join(tran_fig_dir,'snr-yz-repeat%02d.png'%repeat))

    if 1:
        from adcpy import adcpy, adcpy_recipes
        # Show secondary circulation based on ADCPy
        # Note that adcpy wants z to be positive down, starting at the surface.
        trans=[ds_to_adcpy(ds) for ds in tran_dss]

        for t in trans:
            t.lonlat_to_xy("EPSG:26910")

        # if len(trans)>1:
        if 1:
            dx=np.diff(tran_dss[0].x_sample.values)
            dy=np.diff(tran_dss[0].y_sample.values)
            dxy=np.median(np.sqrt(dx**2 + dy**2))
            dxy=max(dxy,2.0) # 2.0m decent for ADCP. May scale larger for model.
            dz=np.nanmedian( np.abs(np.diff(tran_dss[0].z_ctr.values)))
            dz=max(dz,0.1)
            tran_avg=adcpy_recipes.average_transects(trans,dxy=dxy,dz=dz,
                                                     plotline_orientation='river')
        else:
            tran_avg=trans[0] # doesn't work -- some error in ADCPy.

        roz_tran=adcpy_recipes.transect_rotate(tran_avg,rotation='Rozovski')
        fig=adcpy.plot.plot_flow_summary(roz_tran,fig=4)

        map_ax,u_ax,u_cax,v_ax,v_cax = fig.axes
        u_ax.images[0].set_cmap('CMRmap_r')
        u_ax.images[0].set_clim([0,1.2])

        v_ax.images[0].set_cmap('seismic')
        v_ax.images[0].set_clim([-0.2,0.2])

        # can't easily plot bathy, since the coordinate system has been offset.
        bathy().plot(ax=map_ax,cmap='gray',vmin=-20,vmax=6,
                     offset=[-np.nanmin(np.nanmin(roz_tran.xy[:,0])),
                             -np.nanmin(np.nanmin(roz_tran.xy[:,1]))])
        fig.savefig(os.path.join(tran_fig_dir,'velocity-rozovski.png'))

        with open(transect_xy_fn,'at') as fp:
            np.savetxt(fp,tran_avg.xy,delimiter=',')

if 1:
    # Plan view of all transects
    fig=plt.figure(5)
    fig.clf()
    fig.set_size_inches((8,6),forward=True)
    ax=fig.add_axes([0,0,1,1])

    for transect in transects:
        tran_dss=read_transect(transect)

        for ds in tran_dss:
            ax.plot( ds.x_sample.values, ds.y_sample.values, 'r-')

        ax.text( tran_dss[0].x_sample.values[0],
                 tran_dss[0].y_sample.values[0],
                 tran_dss[0].source,ha='right' )

    bathy().plot(ax=ax,cmap='gray',vmin=-20,vmax=6)
    ax.xaxis.set_visible(0)
    ax.yaxis.set_visible(0)

    ax.axis(zoom_overview)
    fig.savefig(os.path.join(fig_dir,"all_transects_map.png"))

##

# Okay - seems like bottom track was thrown off by a bad initial position,
# but overall GPS was okay, and we can correct that track.

# Is it already correct in that mat file?
# Seems like it.
# Seems like there is more data in the mat file overall.
# Should switch to processing the mat file, but format xr dataset in
# the same way.

# Rozovski:
#   8th (I think) has reversed x-axis.  Should make x-axis consistent with
#      the rotation.
