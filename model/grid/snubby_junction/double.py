from stompy.grid import unstructured_grid
import six
six.moves.reload_module(unstructured_grid)
g=unstructured_grid.UnstructuredGrid.read_ugrid('snubby-08-edit60.nc')



gr=g.global_refine(l_thresh=0.25)
##
plt.figure(2).clf()
gr.plot_edges(color='k',lw=0.5)
gr.plot_cells(color='0.8')
##
gr.write_ugrid('snubby-08-refine.nc')
