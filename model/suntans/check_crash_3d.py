from stompy.grid import unstructured_grid
import xarray as xr
import matplotlib.pyplot as plt
##
ds=xr.open_dataset('runs/steady_007/Estuary_SUNTANS.nc.nc.2')
g=unstructured_grid.UnstructuredGrid.from_ugrid(ds)
##
plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

g.plot_edges(ax=ax,lw=0.5,color='k')
ax.axis('equal')

ax.plot(ds.xv.values, ds.yv.values,'g.')

# crash at x=6.4719481e+05, y=4.1858762e+06
# nothing weird in the grid at that location.
#

##

ds.
