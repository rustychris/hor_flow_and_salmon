"""
For each of the transects, summarize the data in its raw form.

Second implementation, using xarray datasets and more flexible
vertical representation
"""
from __future__ import print_function
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import six

from stompy import xr_transect, utils, filters
from stompy.memoize import memoize

##

def sym_clim(coll):
    vals=coll.get_array()
    m=np.nanmax(np.abs(vals))
    clim=[-m,m]
    coll.set_clim(clim)
    return clim

@memoize()
def bathy():
    # from stompy.spatial import field
    #return field.GdalGrid('../../bathy/OldRvr_at_SanJoaquinRvr2012_0104-utm-m_NAVD.tif')
    utils.path(os.path.join( os.path.dirname(__file__),"../../bathy"))
    from bathy import dem
    study_zoom=[646000, 649000, 4185000, 4186500]
    tile=dem().extract_tile(study_zoom)
    return tile

def tran_zoom(ds,fac=0.1):
    zoom=utils.expand_xxyy([np.nanmin(ds.x_sample.values), np.nanmax(ds.x_sample.values),
                              np.nanmin(ds.y_sample.values), np.nanmax(ds.y_sample.values) ],
                             fac)
    return zoom

def set_bounds(ax,ds,fac=0.1):
    # in some cases MPL goes nuts when trying to manipulate
    # datalim.  just give in, let it manipulate the box.
    ax.set_adjustable('box')
    ax.set_aspect(1)
    zoom=tran_zoom(ds,fac=fac)

    # try to keep the aspect ratio the same
    xxyy=ax.axis()
    # dx/dy
    ax_aspect=(xxyy[1]-xxyy[0])/(xxyy[3]-xxyy[2])

    # dx/dy
    dx=(zoom[1]-zoom[0])
    dy=(zoom[3]-zoom[2])
    zoom_aspect=dx/dy
    if zoom_aspect> ax_aspect:
        # expand y
        mid_y=0.5*(zoom[2]+zoom[3])
        zoom[2]=mid_y-0.5*dx/ax_aspect
        zoom[3]=mid_y+0.5*dx/ax_aspect
    else:
        # expand x
        mid_x=0.5*(zoom[0]+zoom[1])
        zoom[0]=mid_x-0.5*dy*ax_aspect
        zoom[1]=mid_x+0.5*dy*ax_aspect

    ax.axis(zoom)

##

def summarize_transect(tran,num=None,w_scale=1.0,quiver_count=75,
                       plot_averages=False,smooth_samples=20,
                       do_quiver=True):
    """
    Given xr_transect formatted dataset tran (i.e. already averaged
    from multiple transects if ADCP), plot an overview figure.
    w_scale: 1. to include w in lateral quiver, 0. to ignore

    plot_averages: include a panel with line plots of the depth-averaged
      velocity.

    smooth_samples: for averages, smooth to get roughly this number of degrees
      of freedom.
    """

    fig=plt.figure(num)
    fig.set_size_inches((8,10),forward=True)
    fig.clf()

    uv_avg=xr_transect.depth_avg(tran,'U')

    nrows=3
    if plot_averages:
        nrows+=1
        ax_avg=fig.add_subplot(nrows,1,nrows)
    else:
        ax_avg=None

    ax_dem=fig.add_subplot(nrows,2,1)
    ax_dem_out=fig.add_subplot(nrows,2,2)

    for ax in [ax_dem,ax_dem_out]:
        bathy().plot(ax=ax,cmap='gray',aspect=1.0)
        ax.xaxis.set_visible(0)
        ax.yaxis.set_visible(0)

    ax_dem.quiver( tran.x_sample.values,
                   tran.y_sample.values,
                   uv_avg[:,0],uv_avg[:,1],
                   color='r')
    ax_dem_out.plot(tran.x_sample.values, tran.y_sample.values,'r-')
    set_bounds(ax_dem,tran)
    set_bounds(ax_dem_out,tran,10)

    xr_transect.add_rozovski(tran)

    ax1=fig.add_subplot(nrows,1,2)
    ax2=fig.add_subplot(nrows,1,3)

    axs=[ax1,ax2]

    pos_dir=tran.z_ctr.attrs.get('positive','up').lower()
    if pos_dir=='down':
        z_sgn=-1
    else:
        z_sgn=1

    if 0: # east/north
        coll_1=xr_transect.plot_scalar_polys(tran,'Ve',ax=ax1,cmap='PuOr')
        coll_2=xr_transect.plot_scalar_polys(tran,'Vn',ax=ax2,cmap='YlGnBu')
        sym_clim(coll_1) ; sym_clim(coll_2)

    if 1: # downstream/left
        coll_1=xr_transect.plot_scalar_polys(tran,tran.Uroz.isel(roz=0),ax=ax1,cmap='CMRmap_r')
        coll_2=xr_transect.plot_scalar_polys(tran,tran.Uroz.isel(roz=1),ax=ax2,cmap='Spectral')
        # consistent with previous plots
        coll_1.set_clim([0,1.2])
        coll_2.set_clim([-0.2,0.2])
        d,z,u,w=xr.broadcast(tran.d_sample,tran.z_ctr,tran.Uroz.isel(roz=1),tran.Vu)

        if do_quiver:
            # negate u here, because in the downstream/left coordinate frame, left is positive,
            # but in the axes, left is negative
            x_vals=d.values.ravel()
            y_vals=z_sgn*z.values.ravel()
            u_vals=-u.values.ravel()
            v_vals=w_scale*w.values.ravel()

            N=len(x_vals)
            if N>quiver_count:
                if 0: # boring stride based
                    stride=int(N/quiver_count)
                    sel=slice(None,None,stride)
                else: # random.
                    sel=np.random.choice(N,size=quiver_count,replace=False)
            else:
                sel=slice(None)

            # go to bitmask so we can mask out nan values too.
            mask=np.zeros(N,np.bool8)
            mask[sel]=True
            mask[ np.isnan(y_vals) ] = False
            sel=mask
            
            quiv=ax2.quiver(x_vals[sel],y_vals[sel],u_vals[sel],v_vals[sel],scale=2.5,angles='xy')
            # Not sure why this necessary, but it seems to be.
            slice_axis=[np.nanmin(x_vals),np.nanmax(x_vals),
                        np.nanmin(y_vals),max(0,np.nanmax(y_vals))]

            ax2.axis(slice_axis)
            ax1.axis(slice_axis)
            ax1.text(0.05,0.05,"Streamwise",transform=ax1.transAxes)
            ax2.text(0.05,0.05,"Lateral",transform=ax2.transAxes)
            ax1.set_facecolor('0.85')
            ax2.set_facecolor('0.85')

    for ax in [ax1,ax2]:
        if 'z_bed' in tran:
            ax.plot(tran.d_sample, z_sgn*tran.z_bed, 'k-', lw=0.8)

    plt.colorbar(coll_1,ax=ax1,label='m/s')
    plt.colorbar(coll_2,ax=ax2,label='m/s')

    fig.subplots_adjust(top=0.98,bottom=0.15,left=0.06,right=0.98)

    if plot_averages:
        pos=ax_avg.get_position()
        ax1_pos=ax1.get_position() # copy width of axes
        pos=[pos.xmin,pos.ymin,ax1_pos.width,pos.height]
        ax_avg.set_position(pos)

        ax_avg.text(0.05,0.05,"Depth average",transform=ax_avg.transAxes)

        u_avg=xr_transect.depth_avg(tran,tran.Uroz.isel(roz=0))

        lp_win=int(len(tran.d_sample)/smooth_samples)
        if lp_win>1:
            def smooth(x): return filters.lowpass_fir(x,winsize=lp_win)
        else:
            def smooth(x): return x

        ax_avg.plot( tran.d_sample, smooth(u_avg.values), label='Streamwise')
        if 'mean_water_speed' in tran: # see if we get something similar to them
            ax_avg.plot( tran.d_sample, smooth(tran.mean_water_speed), label='Mean water speed')
        if 1: # secondary strength
            ax_sec=ax_avg.twinx()
            ax_sec.set_position(pos)

            if 'secondary' not in tran:
                xr_transect.calc_secondary_strength(tran,name='secondary')
            style=dict(lw=0.5, color='g',label='Secondary')
            #ax_sec.plot(tran.d_sample, smooth(tran.secondary), **style)
            ax_sec.fill_between(tran.d_sample, 0, smooth(tran.secondary),
                                alpha=0.3, **style)
            #ax_avg.plot([np.nan],[np.nan],**style)
            ax_sec.axis(ymin=-0.1,ymax=0.1)
            ax_sec.legend(loc='upper right')
        ax_avg.axis(ymin=0)
        ax_avg.legend(loc='upper left')

    Q=xr_transect.Qleft(tran)
    A=xr_transect.total_int(tran,1)
    Ustream_mean=Q/A

    lines=["Q=%.2f m3 s-1"%Q,
           "A=%.2f m2"%A,
           "U streamwise=%.3f m s-1"%Ustream_mean,
           "Source: %s"%getattr(tran,'source','n/a'),
           "Transect: %s"%getattr(tran,'transect_name','n/a')]
    fig.text(0.1,0.11,"\n".join(lines),va='top')

    return fig


##
if __name__=='__main__':
    # testing:
    # for now, pull transect xy from the existing untrim sections
    hydro_txt_fn="../../model/untrim/ed-steady/section_hydro.txt"
    transect_defs=[]
    for name in xr_transect.section_hydro_names(hydro_txt_fn):
        untrim_ds=xr_transect.section_hydro_to_transect(hydro_txt_fn,name)
        line_xy=np.c_[untrim_ds.x_sample.values,untrim_ds.y_sample.values]
        transect_defs.append( dict(name=name,xy=line_xy) )

    source_name="sun_steady_008dt1"
    run_path=source.replace('sun_','../../model/suntans/runs/')
    fig_dir=os.path.join(run_path,"figs-20180830")
    os.path.exists(fig_dir) or os.mkdir(fig_dir)

    from stompy.model.suntans import sun_driver

    sun_model=sun_driver.SuntansModel.load(os.path.join(run_path,'suntans.dat'))

    for fig_i,tran_def in enumerate(transect_defs):
        ds=sun_model.extract_transect(time=-1,xy=tran_def['xy'],dx=3)
        if xr_transect.Qleft(ds)<0:
            # print("flipping %d"%fig_i)
            ds=sun_model.extract_transect(time=-1,xy=tran_def['xy'][::-1,:],dx=3)

        xy=np.c_[ds.x_sample.values,ds.y_sample.values]
        ds.attrs['source']=source_name
        ds.attrs['transect_name']=tran_def['name']

        fig=summarize_transect(ds,w_scale=0.0, num=20+fig_i)

        fig.savefig(os.path.join(fig_dir,'transect_summary-%s.png'%tran_def['name']),
                    dpi=150)


