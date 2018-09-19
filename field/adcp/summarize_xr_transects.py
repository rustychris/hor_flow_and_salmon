"""
For each of the transects, summarize the data in its raw form.

Second implementation, using xarray datasets and more flexible
vertical representation
"""
from __future__ import print_function
from stompy import xr_transect
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
    utils.path("../../bathy")
    from bathy import dem
    study_zoom=[646000, 649000, 4185000, 4186500]
    tile=dem().extract_tile(study_zoom)
    return tile

def tran_zoom(ds,fac=0.1):
    return utils.expand_xxyy([ds.x_sample.values.min(), ds.x_sample.values.max(),
                              ds.y_sample.values.min(), ds.y_sample.values.max() ],
                             fac)

def set_bounds(ax,ds,fac=0.1):
    ax.axis('equal')
    ax.axis(tran_zoom(ds,fac=fac))

##

def summarize_transect(tran,num=None,w_scale=1.0):
    """
    Given xr_transect formatted dataset tran (i.e. already averaged
    from multiple transects if ADCP), plot an overview figure.
    w_scale: 1. to include w in lateral quiver, 0. to ignore
    """

    fig=plt.figure(num)
    fig.set_size_inches((8,10),forward=True)
    fig.clf()

    uv_avg=xr_transect.depth_avg(tran,'U')

    ax_dem=fig.add_subplot(3,2,1)
    ax_dem_out=fig.add_subplot(3,2,2)

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

    ax1=fig.add_subplot(3,1,2)
    ax2=fig.add_subplot(3,1,3)

    axs=[ax1,ax2]

    if 0: # east/north
        coll_1=xr_transect.plot_scalar_polys(tran,'Ve',ax=ax1,cmap='PuOr')
        coll_2=xr_transect.plot_scalar_polys(tran,'Vn',ax=ax2,cmap='YlGnBu')
        sym_clim(coll_1) ; sym_clim(coll_2)

    if 1: # downstream/left
        coll_1=xr_transect.plot_scalar_polys(tran,tran.Uroz.isel(roz=0),ax=ax1,cmap='CMRmap_r')
        coll_2=xr_transect.plot_scalar_polys(tran,tran.Uroz.isel(roz=1),ax=ax2,cmap='seismic')
        # consistent with previous plots
        coll_1.set_clim([0,1.2])
        coll_2.set_clim([-0.2,0.2])
        d,z,u,w=xr.broadcast(tran.d_sample,tran.z_ctr,tran.Uroz.isel(roz=1),tran.Vu)

        # negate u here, because in the downstream/left coordinate frame, left is positive,
        # but in the axes, left is negative
        quiv=ax2.quiver(d,z,-u,w_scale*w,scale=2.5,angles='xy')
        ax1.text(0.05,0.05,"Streamwise",transform=ax1.transAxes)
        ax2.text(0.05,0.05,"Lateral",transform=ax2.transAxes)

    plt.colorbar(coll_1,ax=ax1,label='m/s')
    plt.colorbar(coll_2,ax=ax2,label='m/s')

    fig.subplots_adjust(top=0.98,bottom=0.15,left=0.06,right=0.98)

    Q=xr_transect.Qleft(tran)
    A=xr_transect.total_int(tran,1)
    Ustream_mean=Q/A

    lines=["Q=%.2f m3 s-1"%Q,
           "A=%.2f m2"%A,
           "U streamwise=%.3f m s-1"%Ustream_mean,
           "Source: %s"%ds.source,
           "Transect: %s"%getattr(ds,'transect_name','n/a')]
    fig.text(0.1,0.11,"\n".join(lines),va='top')
    return fig


# fig=summarize_transect(ds,num=10)


##

# testing:
# for now, pull transect xy from the existing untrim sections
hydro_txt_fn="../../model/untrim/ed-steady/section_hydro.txt"
transect_defs=[]
for name in xr_transect.section_hydro_names(hydro_txt_fn):
    untrim_ds=xr_transect.section_hydro_to_transect(hydro_txt_fn,name)
    line_xy=np.c_[untrim_ds.x_sample.values,untrim_ds.y_sample.values]
    transect_defs.append( dict(name=name,xy=line_xy) )

##

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


