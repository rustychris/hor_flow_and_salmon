import matplotlib.pyplot as plt
import xarray as xr
from stompy.model.delft import dfm_grid
from stompy.plot import plot_utils

##

ds=xr.open_dataset('runs/hor_002/DFM_OUTPUT_flowfm/flowfm_20120801_000000_map.nc')

g=dfm_grid.DFMGrid(ds)

##

# Free surface - looks fine.
plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

ccoll=g.plot_cells(ax=ax,values=ds.s1.isel(time=-1),cmap='jet')
ccoll.set_clim([1.9,2.2])

plot_utils.cbar(ccoll)
ax.axis('equal')

##

# Velocity magnitude
plt.figure(2).clf()
fig,ax=plt.subplots(num=2)

ucxa=ds.ucxa.isel(time=-1).values
ucya=ds.ucya.isel(time=-1).values

vmag=np.sqrt(ucxa**2 + ucya**2)

ccoll=g.plot_cells(ax=ax,values=vmag,cmap='jet')
plot_utils.cbar(ccoll)
ax.axis('equal')

# Max out around 0.4 m/s, with more like 0.2 m/s at the junction.

##

# Is it at steady state?

# upstream of junction
xy=(647300, 4185800)
c=g.select_cells_nearest( xy )

c_ucxa=ds.ucxa.isel(nFlowElem=c).values
c_ucya=ds.ucya.isel(nFlowElem=c).values

c_mag=np.sqrt(c_ucxa**2 + c_ucya**2)

plt.figure(3).clf()
fig,ax=plt.subplots(num=3)
ax.plot(ds.time.values, c_mag)






