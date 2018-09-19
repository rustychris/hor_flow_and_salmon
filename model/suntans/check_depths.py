plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

ccoll=model.grid.plot_cells(values=model.grid.cells['cell_depth'],ax=ax)
plt.colorbar(ccoll)
ccoll.set_cmap('jet')
ax.axis('equal')

# I'm setting a freesurface BC around

