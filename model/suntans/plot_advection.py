ds=xr.open_dataset('runs/steady_011B/Estuary_SUNTANS.nc.nc.2')
g=unstructured_grid.UnstructuredGrid.from_ugrid(ds)

##

# banding is strongest near the surface
ctop=5
u=ds.uc.isel(Nk=ctop,time=-1)
v=ds.vc.isel(Nk=ctop,time=-1)

umag=np.sqrt(u**2+v**2)

plt.figure(1).clf()

coll=g.plot_cells(values=umag,cmap='jet')
plt.axis('equal')
coll.set_clim([0.4,0.6])
plt.colorbar(coll)
zoom=(647161.8866876907, 647390.5969737027, 4185736.0783451973, 4185949.1109099905)
plt.axis(zoom)

cc=g.cells_center()
sel=g.cell_clip_mask(zoom)
plt.quiver( cc[sel,0],cc[sel,1],u[sel],v[sel],scale=5 )

U=ds.U.isel(Nk=ctop,time=-1)
sel=g.edge_clip_mask(zoom)
ec=g.edges_center()
normals=-g.edges_normals()
if 0:
    plt.quiver(ec[sel,0],ec[sel,1],
               (U*normals[:,0])[sel],(U*normals[:,1])[sel],
               color='m',scale=5)
if 1:
    plt.quiver(ec[sel,0],ec[sel,1],
               normals[:,0][sel],normals[:,1][sel],
               color='m',scale=15)

##

# Cells where the stripe is getting stronger:
#   +cell:
#       2x longitudinal in, inner bend in, outer bend out
#   -cell:
#       2x longitudinal out, inner in, outer out
# hmmm -- not super obvious pattern looking at a few others
