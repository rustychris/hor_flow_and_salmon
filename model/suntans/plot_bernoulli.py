import xarray as xr

##
#ds_adv=xr.open_dataset('runs/straight_03/Estuary_SUNTANS.nc.nc.0')
#ds_noa=xr.open_dataset('runs/straight_04/Estuary_SUNTANS.nc.nc.0')
ds_bend=xr.open_dataset('runs/bend_00/Estuary_SUNTANS.nc.nc.0')

##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

to_plot=[ (ds_bend,'-','2D') ]
# to_plot= [(ds_adv,'-','adv'),
#           (ds_noa,'--','noa')]

for ds,ls,label in to_plot:
    x=ds.xv.values
    ord=np.argsort(x)

    # Eta
    ax.plot( x[ord], ds.eta.isel(time=-1).values[ord],color='r',ls=ls,label=label+' eta')

    # bed
    ax.plot( x[ord], -ds.dv.values[ord], color='k',ls=ls, label=label+'bed')

    # bernoulli
    eta=ds.eta.isel(time=-1).values
    u=ds.uc.isel(time=-1,Nk=0).values
    bern=eta + u**2/(2*9.8)
    ax.plot( x[ord], bern,color='b',ls=ls,label=label+' bern')

ax.legend()
