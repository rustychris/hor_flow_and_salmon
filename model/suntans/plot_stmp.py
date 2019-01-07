import stompy.plot.cmap as scmap
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from stompy import utils
from stompy.model.suntans import sun_driver

run_dir="runs/snubby_056r"
model=sun_driver.SuntansModel.load(run_dir)

##

fig=plt.figure(1,figsize=(6.4,4.8))
fig.clf()
fig,axs=plt.subplots(2,1,num=1,sharex=True,sharey=True)
ax=axs[0]
Qs=[]

ti=-1
#for proc in range(model.num_procs):
for proc in [0]:
    g=model.subdomain_grid(proc)

    fn=os.path.join(run_dir,"stmp.dat.%d"%proc)

    stmps=np.fromfile(fn,np.float64)
    stmps=stmps.reshape( (-1,g.Ncells(), int(model.config['Nkmax']), 2) )
    stmps[stmps==-999]=np.nan
    stmp=stmps[ti,...]
    
    g.plot_edges(lw=0.2,color='b',ax=ax)

    cc=g.cells_center()
    sel=np.isfinite(stmp[:,0,0])

    Q=ax.quiver(cc[sel,0],cc[sel,1],-stmp[sel,0,0],-stmp[sel,0,1],
                scale=0.1)
    Qs.append(Q)

plt.setp(ax.get_xticklabels(),visible=0)
plt.setp(ax.get_yticklabels(),visible=0)
ax.quiverkey(Qs[0],0.1,0.1,1e-2,"1e-2 m/s$^2$")
fig.tight_layout()
ax.axis('equal')
#ax.axis((647322, 647723., 4185445., 4185739))
zoom=(647389.936895501, 647537.7026456677, 4185566.151332026, 4185693.5559700294)
#zoom=(647430.0528457537, 647470.6089196375, 4185602.8215450295, 4185638.718048039)
ax.axis(zoom)

ds=xr.open_dataset(model.map_outputs()[proc])

#coll=g.plot_cells(labeler=lambda i,r: "%.2f"%(ds.dzz.values[ti,0,i]),clip=zoom,ax=ax)
#coll.set_visible(0)
plt.setp(ax.texts,color='b')

#ecoll=g.plot_edges(labeler=lambda i,r: "%.2f"%(ds.dzf.values[ti,0,i]),clip=zoom,ax=ax)
#ecoll.set_visible(0)
plt.setp(ax.texts,ha='center',va='center',clip_on=True)


#ecoll2=g.plot_edges(labeler=lambda j,r: str(j),ax=axs[1],clip=zoom)
ecoll2=g.plot_edges(ax=axs[1],clip=zoom)
#ccoll2=g.plot_cells(labeler=lambda i,r: str(i),ax=axs[1],clip=zoom)
#ccoll2.set_visible(0)
#plt.setp(axs[1].texts,ha='center',va='center',clip_on=True)

sel=g.cell_clip_mask(zoom)
Q=axs[1].quiver(ds.xv.values[sel],ds.yv.values[sel],
                ds.uc.values[ti,0,sel],ds.vc.values[ti,0,sel],
                scale=10,color='orange')

ax.axis((647428.642495636, 647445.8612984748, 4185620.445806887, 4185631.398008547))
##

# Look at 530: how does it get lateral?
#   the sides have velocity of 8mm/s and 16 mm/s.
#   compared to .57 and 0.59 longitudinal.
# downstream advection term:
#   upstream edge is shallower, in this case about 7% shallower.
#   
def cell_gradient_of_edges(self,edge_vals):
    cell_side_gradients=np.zeros( (self.Ncells(),2), np.float64)
    ec=self.edges_center()
    cc=self.cells_center()
    for c in range(self.Ncells()):
        pnts=[]
        for j in self.cell_to_edges(c):
            pnts.append( [ec[j,0]-cc[c,0],
                          ec[j,1]-cc[c,1],
                          edge_vals[j]] )
        pnts=np.array(pnts)
        cell_side_gradients[c,0],_=np.polyfit(pnts[:,0],pnts[:,2],1)
        cell_side_gradients[c,1],_=np.polyfit(pnts[:,1],pnts[:,2],1)
    return cell_side_gradients

grad_dzf = cell_gradient_of_edges(g,ds.dzf.values[ti,0,:])

u_unit=np.c_[ds.uc[ti,0,:],
             ds.vc[ti,0,:]]
u_unit=utils.to_unit(u_unit)
u_unit[np.isnan(u_unit)]=0.0
dzf_dot_u=(grad_dzf[:,0]*u_unit[:,0] + grad_dzf[:,1]*u_unit[:,1])

#Qgrad=ax.quiver(cc[sel,0],cc[sel,1],grad_dzf[sel,0],grad_dzf[sel,1],
#                color='r')
ccoll=g.plot_cells(ax=ax,values=dzf_dot_u)
ccoll.set_zorder(-5)
ccoll.set_cmap('seismic')
ccoll.set_clim([-0.1,0.1])
##
if 0:
    gg=unstructured_grid.UnstructuredGrid.read_suntans(run_dir)
    depth=np.loadtxt(os.path.join(run_dir,"depths.dat-voro"))
    ccoll=gg.plot_cells(values=depth[:,2],zorder=-2)
    cmap=scmap.load_gradient('ncview_banded.cpt')
    ccoll.set_cmap(cmap)

##
