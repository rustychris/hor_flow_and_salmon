import matplotlib.pyplot as plt
import xarray as xr

from stompy.model.delft import dfm_grid


##


ds=xr.open_dataset('runs/hor_004-restart2/DFM_OUTPUT_flowfm/flowfm_0001_20120801_013000_map.nc')
g=dfm_grid.DFMGrid(ds)

##

# zoom=(648785, 648972., 4184394., 4184568)
zoom=(648731., 648999, 4184341., 4184579)
plt.figure(20).clf()

g.plot_edges(lw=0.5,color='k',clip=zoom)
plt.axis('equal')

x=ds.FlowLink_xu.values
y=ds.FlowLink_yu.values

wdim=1
nu_t=ds.vicwwu.isel(wdim=wdim,time=1).values

from matplotlib import colors

scat=plt.scatter(x,y,20,nu_t,
                 norm=colors.LogNorm(1e-6,0.1,clip=True),
                 cmap='jet')

# Site of the instability:
plt.plot( [648867],[4.18449e+06],'kx')

plt.colorbar(scat,label=r'$\nu_T$ m$^2$ s$^{-1}$')

#plt.axis(zoom)
plt.axis( (648838, 648899, 4184462, 4184516) )

plt.title("Eddy viscosity, wdim=%d"%wdim)
