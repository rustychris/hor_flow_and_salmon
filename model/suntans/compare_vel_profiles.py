"""
Trying to understand vertical velocity profiles

"""

# transect 3 is maybe the best for comparisons.
adcp_ds=xr.open_dataset("../../field/adcp/040518_BT/040518_3_BTref-avg.nc")
adcp_xy=np.c_[adcp_ds.x_sample.values,adcp_ds.y_sample.values]

##

run_dirs=["runs/steady_013",
          "runs/steady_014",
          "runs/steady_015",
          "runs/steady_016",
          "runs/steady_017"
]

model_dss=[]
for rd in run_dirs:
    print(rd)
    model=sun_driver.SuntansModel.load(rd)
    ds_orig=model.extract_transect(xy=adcp_xy[ [0,-1], :],
                                   time=-1,dx=2)
    wet_samples=np.nonzero(np.abs(ds_orig.z_dz.sum(dim='layer').values)>0.01)[0]
    sample_slice=slice(wet_samples[0],wet_samples[-1]+1)
    ds=ds_orig.isel(sample=sample_slice)
    ds.attrs.update(ds_orig.attrs)

    xr_transect.get_d_sample(ds)

    # shift sun_ds to get 0=fs
    z_shift=float(ds.eta.mean())
    ds.z_ctr.values[:]-=z_shift
    ds.z_int.values[:]-=z_shift
    ds.eta.values[:]-=z_shift

    ds.close()
    model_dss.append(ds)

##
if 0:
    fig_adcp=summarize_xr_transects.summarize_transect(adcp_ds,num=2,w_scale=0.0)
    fig_suns=[]
    for i,ds in enumerate(model_dss):
        fig=summarize_xr_transects.summarize_transect(ds,num=3+i,w_scale=0.0)
        fig.suptitle(ds.source)
        fig_suns.append(fig)

##

slice_ds=[]
for ds in [adcp_ds]+model_dss:
    # subselect just the center:
    if 'Uroz' not in ds:
        xr_transect.add_rozovski(ds)

    sel=utils.within(ds.d_sample,[30,40])
    sub_ds=ds.isel(sample=sel)
    if sub_ds.z_ctr.ndim>1:
        new_z=np.arange(sub_ds.z_ctr.min(),sub_ds.z_ctr.max(),0.1)
        sub_ds=xr_transect.resample_z(sub_ds,new_z)
    sub_ds['Uroz_prof']=sub_ds.Uroz.mean(dim='sample')
    sub_ds.attrs.update(ds.attrs)
    slice_ds.append(sub_ds)

##

plt.figure(10).clf()
fig,ax=plt.subplots(num=10)

for i,ds in enumerate(slice_ds):
    z=ds.z_ctr.values
    if ds.z_ctr.attrs.get('positive','up')=='down':
        z=-z
    ax.plot(ds.Uroz_prof.isel(roz=0),z,
            label=ds.attrs.get('source',str(i)))

    if 'depth_bt' in ds:
        depth=-float(ds.depth_bt.mean())
        ax.axhline(depth,color='k')
ax.legend()

##

six.moves.reload_module(xr_transect)

def flip_vertical(ds):
    ds=ds.isel(layer=slice(None,None,-1)).copy()
    if ds.z_ctr.attrs.get('positive','up') =='up':
        newpos='down'
    else:
        newpos='up'

    for v in ['z_ctr','z_dz','z_int']:
        if v in ds:
            ds[v].values[:] *= -1
            ds[v].attrs['positive']=newpos

    return ds
fadcp_ds=flip_vertical(adcp_ds)

##

# this doesn't work yet
#com_adcp,com_sun = xr_transect.resample_to_common( [fadcp_ds,sun_ds] )
#fig_cadcp=summarize_xr_transects.summarize_transect(com_adcp,num=2,w_scale=0.0)
#fig_csun  =summarize_xr_transects.summarize_transect(com_sun, num=3,w_scale=0.0)


