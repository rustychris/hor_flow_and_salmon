from stompy.grid import unstructured_grid, triangulate_hole, exact_delaunay, front, shadow_cdt
import six
six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(shadow_cdt)
six.moves.reload_module(front)
six.moves.reload_module(triangulate_hole)


##

g=unstructured_grid.UnstructuredGrid.read_pickle('merge_v16_edit06.pkl')


##
# Fails b/c the scale is way way big, or too rigid
seed=[648075.577031672, 4185627.979154055]

g=triangulate_hole.triangulate_hole(g,seed_point=seed,apollo_rate=1.15)

g.write_pickle('merge_v16_edit07.pkl')

##

plt.figure(1).clf()
g.plot_edges(lw=0.5) # clip=clip) #,labeler='id')

# All of the edges are RIGID, but two of them shouldn't be.
# That's fixed, but it's still failing..
#AT.grid.plot_edges(values=AT.grid.edges['fixed'],cmap='rainbow',lw=3)
#AT.grid.plot_nodes(values=AT.grid.nodes['fixed'],cmap='rainbow',sizes=40)

plt.axis('equal')

##

# j: 35540, 35541

