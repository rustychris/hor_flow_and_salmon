six.moves.reload_module(summarize_xr_transects)

tran_nc_fn="040518_BT/040518_5BTref-avg.nc"
# tran_nc_fn="040518_BT/040518_7_BTref-avg.nc"
ds=xr.open_dataset(tran_nc_fn)
fig=summarize_xr_transects.summarize_transect(ds,num=20,w_scale=0.0,plot_averages=True)

# would like some measure of the strength of the secondary circulation, per-water column.

##



