from stompy.spatial import field

from stompy.grid import unstructured_grid

g=unstructured_grid.UnTRIM08Grid('junction_29_w_depth_2007.grd')

g.delete_cell_field('subgrid')
g.delete_edge_field('subgrid')

##

# There is too much weird stuff in the depth data on that grid.
# Start over from DEMs

mrf=field.MultiRasterField(["/Users/rusty/data/bathy_dwr/gtiff/*.tif"])


##

edge_xy=g.edges_center()


g.add_edge_field('depth',mrf(edge_xy))


##

g.add_node_field('depth',mrf(g.nodes['x']))

##

from stompy.grid import depth_connectivity
six.moves.reload_module(depth_connectivity)


node_depth=depth_connectivity.greedy_edge_mean_to_node(g,edge_depth=edge_depth)


##

plt.figure(10).clf()
ecoll=g.plot_edges(values=g.edges['depth_mean'])
plt.colorbar(ecoll)
plt.axis('equal')

## 
g.write_ugrid('derived/grid_ugrid.nc',overwrite=True)
##

dfm_grid.write_dfm(g,'derived/grid_net.nc',overwrite=True)
##
