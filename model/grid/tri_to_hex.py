import os
from stompy.grid import unstructured_grid

g_tri=unstructured_grid.SuntansGrid('junction-grid-100')

##

g_hex=g_tri.create_dual(center='circumcenter',create_cells=True)


##

plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

g_tri.plot_edges(ax=ax,color='b',lw=0.3)
g_hex.plot_edges(ax=ax,color='g',lw=0.3)
g_hex.plot_cells(ax=ax,alpha=0.2)

ax.axis('equal')

# checking on orthogonality at a problem spot:
zoom=(647275.5263519865, 647295.9225515076, 4185924.1187208737, 4185934.4744617515)
ax.axis(zoom)

cc_hex=g_hex.cells_center()
mask=g_hex.cell_clip_mask(zoom)
ax.plot(  cc_hex[mask,0], cc_hex[mask,1], 'bo')

##

output_dir="junction-grid-101"
os.path.exists(output_dir) or os.mkdir(output_dir)

g_hex.write_suntans_hybrid(output_dir,overwrite=True)

