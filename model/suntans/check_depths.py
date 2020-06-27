from stompy.model.suntans import sun_driver

## 
model=sun_driver.SuntansModel.load('runs/snubby_062')

g0=model.subdomain_grid(2)

j_xyz=np.loadtxt( os.path.join(model.run_dir,"depths.dat-edge.2") )

## 
plt.figure(1).clf()
fig,ax=plt.subplots(1,1,num=1)

ccoll=g0.plot_cells(values=g0.cells['dv'],ax=ax)
ecoll=g0.plot_edges(values=j_xyz[:,2],lw=2,zorder=2,ax=ax)
ecoll_bg=g0.plot_edges(color='k',lw=3,zorder=1)

plt.colorbar(ccoll)
plt.setp([ecoll,ccoll], clim=[3,6],cmap='jet')

ax.axis('equal')


