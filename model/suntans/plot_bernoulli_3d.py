import xarray as xr
import stompy.grid.unstructured_grid as ugrid

##
#ds_3d=xr.open_dataset('runs/straight_06/Estuary_SUNTANS.nc.nc.0')
ds_3d=xr.open_dataset('runs/straight_01/Estuary_SUNTANS.nc.nc.0')
ds_3d['Nki']=('Nc',), ds_3d.Nk.values

del ds_3d['Nk']
ds_3d['wc']=('time','Nk','Nc'), ds_3d['w'].isel(Nkw=slice(None,-1)).values

##
ds=ds_3d.isel(time=-1)

uc,vc,wc,eta = xr.broadcast(ds.uc,ds.vc,ds.wc,ds.eta)

bernoulli=eta.values + (uc.values**2+vc.values**2 + wc.values**2)/(2*9.8)

## 
_,x,y,z = xr.broadcast(ds.uc,ds.xv,ds.yv,ds.z_r)

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

clim=[np.nanmin(bernoulli),
      np.nanmax(bernoulli)]

for k in range(ds.dims['Nk']):
    seg_x=x[k,:]
    seg_z=(-z+10*bernoulli)[k,:]
    ax.plot(seg_x,seg_z,'k-',lw=0.4)
    scat=ax.scatter(seg_x,seg_z,
                    20,bernoulli[k,:],cmap='jet')

