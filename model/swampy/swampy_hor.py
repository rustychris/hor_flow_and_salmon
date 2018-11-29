from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
import SWAMPpy
import numpy as np
import six

##

six.moves.reload_module(SWAMPpy)

class SwampyHOR(SWAMPpy.SwampyCore):
    def set_grid(self,ug):
        """
        Set grid from unstructured_grid.UnstructuredGrid instance
        """
        self.grd=ug
        self.set_topology()

    def set_initial_conditions(self,h=0.0):
        # freesurface per cell -- allocate
        self.ic_ei=np.zeros(self.ncells, np.float64)
        # bed elevation, positive-down -- allocate
        self.ic_zi=-self.grd.cells['cell_depth']

        # set values
        self.ic_ei[:]=h

        # self.ds_eta=? # eta at downstream BC?
        # set_initial_conditions uses ds_i, us_i, us_j index arrays
        # but are those used outside set initial condition?
        # yes, yes they are.

    fig=None
    ax=None
    def prep_figure(self):
        plt.figure(1).clf()
        self.fig,self.ax=plt.subplots(num=1)
        # self.grd.plot_edges(ax=self.ax,color='k',lw=0.5)
        # self.ax.axis('equal')
        self.ax.xaxis.set_visible(0)
        self.ax.yaxis.set_visible(0)

    def step_output(self,n,ei,**kwargs):
        if self.fig is None:
            return
        if n%5:
            return

        self.prep_figure()
        ccoll=sim.grd.plot_cells(values=sim.hi,cmap='jet',ax=self.ax)
        cc=sim.grd.cells_center()
        ui = sim.get_center_vel(sim.uj)
        quiv=self.ax.quiver(cc[:,0],cc[:,1],ui[:,0],ui[:,1])
        plt.colorbar(ccoll)
        self.ax.quiverkey(quiv,0.1,0.1,0.5,"0.5 m/s")
        self.ax.axis('equal')
        self.fig.canvas.draw()
        plt.pause(0.001)


# not stable for long at dt=2.0
# seems that center_vel is ill-behaved at the water's edge
# crashed at 240s.  trying 0.5, but also clamped cg_tol a bit
# nope -- now it crashes at 273, same way.  
sim=SwampyHOR(dt=0.5)
sim.cg_tol=1e-5 # having a hard time converging.
g=unstructured_grid.UnstructuredGrid.from_ugrid("../suntans/snubby-with_bathy.nc")
g.orient_edges()
sim.set_grid(g)
sim.set_initial_conditions(h=2.2)

sj_inflow=np.array( [[648045.3, 4185562.3],
                     [648035.2, 4185696.1]] )
sj_outflow=np.array( [[647513.1, 4186185.6],
                      [647542.6, 4186070.4]] )
or_outflow=np.array( [[646845.5, 4185762.2],
                      [646857.0, 4185842.4]] )

sim.add_bc( SWAMPpy.FlowBC(geom=sj_inflow,Q=220,ramp_time=1800) ) # 220
sim.add_bc( SWAMPpy.FlowBC(geom=sj_outflow,Q=-100,ramp_time=1800) )
sim.add_bc( SWAMPpy.StageBC(geom=or_outflow,h=2.2) )


# tend probably wrong
sim.prep_figure()
plt.draw()
(hi, uj, tvol, ei) = sim.run(tend=50.)  # final water surface elevation, velocity
ui = sim.get_center_vel(sim.uj)

##

# it continues to get very high velocities at what looks like the
# wetting front.
#
zoom=(647917.6337120306, 648070.6623977675, 4185565.856687871, 4185712.722214185)
csel=sim.grd.cell_clip_mask(zoom)

uih = sim.get_center_vel_hweight(sim.uj,sim.hjbar,sim.hi)
plt.figure(2).clf()
fig,ax=plt.subplots(num=2)
ccoll=sim.grd.plot_cells(values=sim.hi,cmap='jet',ax=ax,mask=csel,
                         labeler=lambda i,r: "%.2f"%sim.ei[i] )
ccoll.set_clim([0,0.2])
cc=sim.grd.cells_center()
ui = sim.get_center_vel(sim.uj)
quiv=ax.quiver(cc[csel,0],cc[csel,1],ui[csel,0],ui[csel,1])
ec=sim.grd.edges_center()
esel=sim.grd.edge_clip_mask(zoom)
ujvec=(sim.uj[:,None]*sim.en)
quiv2=ax.quiver(ec[esel,0],ec[esel,1],ujvec[esel,0],ujvec[esel,1],
                color='r')
ecoll=sim.grd.plot_edges(values=sim.hjbar,cmap='PuOr',ax=ax,
                         mask=esel,lw=3)
ecoll.set_clim([-.001,.001])

plt.colorbar(ccoll)
ax.quiverkey(quiv,0.1,0.1,0.5,"0.5 m/s")
ax.axis('equal')
#ax.axis(zoom)
ax.axis((647969.479350166, 647998.277671164, 4185662.427936126, 4185692.6341643427))
# one such edge:
j=6337

# 20m/s
# hjbar, hjstar, hjtilde all are 0.0
#  hjbar can be positive between a wet and dry cell
#  hjstar seems to stay dry on those edges
# hjtilde seems to be similar to hjbar

# one problem is that those dry cells have ei that are lower
# than the neighboring wet cell.  they are all ei=2.28
# but the bed level is above that.

##

# Stelling paper:
#   hjstar: could be first-order upwinding, and if uj==0,
#       it's max(ei) + min(di), so the higher freesurface
#          and the higher bed elevation.
#          

# eq 9:
#   h-n+1....

# need to ask Ed what paper they were following.
