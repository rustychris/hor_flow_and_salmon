import xarray as xr
import stompy.grid.unstructured_grid as ugrid

##
#ds_adv=xr.open_dataset('runs/straight_03/Estuary_SUNTANS.nc.nc.0')
#ds_noa=xr.open_dataset('runs/straight_04/Estuary_SUNTANS.nc.nc.0')
ds_bend=xr.open_dataset('runs/bend_00/Estuary_SUNTANS.nc.nc.0')

##

plt.figure(1).clf()

to_plot=[ (ds_bend,'-','2D') ]
# to_plot= [(ds_adv,'-','adv'),
#           (ds_noa,'--','noa')]

fig,axs=plt.subplots(len(to_plot),1,sharex=True,sharey=True,num=1)
axs=[axs]

for i,(ds,ls,label) in enumerate(to_plot):
    ax=axs[i]
    g=ugrid.UnstructuredGrid.from_ugrid(ds)

    # Eta
    eta=ds.eta.isel(time=-1)

    # surface values
    uc=ds.uc.isel(time=-1)
    vc=ds.vc.isel(time=-1)
    dzz=ds.dzz.isel(time=-1)

    usurf=np.nan*np.ones(g.Ncells())
    vsurf=np.nan*np.ones(g.Ncells())
    dzmin=0.001
    for c in range(g.Ncells()):
        surf=np.nonzero(dzz.values[:,c]>dzmin)[0]
        if len(surf):
            usurf[c]=uc[surf[0],c]
            vsurf[c]=vc[surf[0],c]

    #g.plot_cells(values=eta.values,ax=ax)
    #cc=g.cells_center()
    #ax.quiver( cc[:,0],cc[:,1], usurf,vsurf)
    bern=(usurf**2+vsurf**2)/(2*9.8) + eta.values
    g.plot_cells(values=bern,ax=ax)


