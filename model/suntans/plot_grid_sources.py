# Compare bathy which is already in some grids, see if any can
# be borrowed.
import matplotlib.pyplot as plt

from stompy.model.delft import dfm_grid
from stompy.grid import unstructured_grid
from stompy.plot import plot_utils

##
r18=dfm_grid.DFMGrid('grid-sources/r18b_net.nc')

r17=dfm_grid.DFMGrid('grid-sources/cascade/r17b_net.nc')

sch=unstructured_grid.UnstructuredGrid.from_ugrid('grid-sources/schout_161.nc')
sch_ds=xr.open_dataset('grid-sources/schout_161.nc')

##
zoom=(608898.0112964935, 631668.6427644436, 4265130.767323407, 4297737.460249973)

plt.figure(20).clf()

fig,axs=plt.subplots(1,3,num=20,sharex=True,sharey=True)

colls=[]
for ax,g,name in zip(axs,
                     [r18,r17,sch],
                     ["r18","r17","sch"]):
    g.plot_edges(ax=ax,lw=0.3,color='k',clip=zoom)
    if 'depth' in g.nodes.dtype.names:
        if name=='sch':
            sgn=-1
        else:
            sgn=1
        colls.append( g.plot_nodes(ax=ax,values=sgn*g.nodes['depth'],clip=zoom,
                                   cmap='jet') )


axs[0].axis('equal')

cax=fig.add_axes( [0.94,0.3,0.02,0.5] )
plot_utils.cbar(colls[0],extras=colls[1:],cax=cax)
fig.subplots_adjust(right=0.9)
