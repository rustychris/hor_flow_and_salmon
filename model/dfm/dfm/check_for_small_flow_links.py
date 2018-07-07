from stompy.grid import unstructured_grid
from stompy.model.delft import dfm_grid

##

# First check the global grid:
g=dfm_grid.DFMGrid('runs/hor_004-restart/grid_net.nc')

##

e2c=g.edge_to_cells(recalc=True)

cc=g.cells_center()

##

nc1=e2c[:,0]
nc2=e2c[:,1]

dists= utils.dist( cc[nc1] - cc[nc2] )
dists[ (nc1<0) | (nc2<0) ] = 100 # np.nan

##

plt.figure(10).clf()
fig,ax=plt.subplots(num=10)

ecoll=g.plot_edges(values=dists,lw=2,ax=ax)
ecoll.set_clim([0,2])

ax.axis('equal')
