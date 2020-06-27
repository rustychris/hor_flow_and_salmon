import xarray as xr
from stompy.grid import ugrid
from stompy.model.suntans import sun_driver

##
model=sun_driver.SuntansModel.load("runs/steady_008dt1/suntans.dat")

##

dss=[xr.open_dataset(fn) for fn in model.map_outputs()]


# 2018-09-15: fixed in suntans code

# for ds in dss:
#     # clean up bad metadata in the suntans output
#     ds.dv.attrs['standard_name']=ds.dv.attrs['stanford_name'] # doh!
#     ds.dv.attrs['positive']='down'
#     ds.eta.attrs['positive']='up'
#     ds.z_r.attrs['positive']='down'

##
zoom=(647014., 647584., 4185599., 4186016)

fig=plt.figure(30)
fig.clf()
ax=fig.add_axes([0,0,1,1])
ax.xaxis.set_visible(0)
ax.yaxis.set_visible(0)

for ds in dss:
    ug=ugrid.UgridXr(ds,
                     face_u_vname='uc',
                     face_v_vname='vc',
                     face_eta_vname='eta')

    uv=ug.get_cell_velocity(time_slice=-1,)
    weights=ug.vertical_averaging_weights(time_slice=-1,
                                          ztop=0,zbottom=0)

    bad=~np.isfinite(uv)
    uv[bad]=0.0

    ua=(uv[:,:,0] * weights).sum(axis=1)
    va=(uv[:,:,1] * weights).sum(axis=1)

    sel=utils.within_2d(np.c_[ds.xv,ds.yv],
                        zoom)
    if not np.any(sel):
        continue
    mags=utils.mag( np.c_[ua[sel],va[sel]] )
    qset=ax.quiver( ds.xv[sel], ds.yv[sel], ua[sel],va[sel], mags,
                    scale=50,cmap='jet')
    qset.set_clim([0,0.8])
ax.axis('equal')
ax.axis(zoom)
#plt.colorbar(qset)

from stompy.plot import plot_utils

#plot_utils.savefig_geo(fig,'velocity-snap.png',transparent=True,dpi=150)


