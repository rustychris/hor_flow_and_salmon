"""
Investigating the difficulty of supporting wetting and drying in swampy.
"""
import stompy.grid.unstructured_grid as ugrid
import matplotlib.pyplot as plt
import SWAMPpy
import numpy as np
from stompy import utils
import six

##

six.moves.reload_module(SWAMPpy)

class SwampyWetDry(SWAMPpy.SwampyCore):
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
        self.ic_ei[:]=np.maximum(h,-self.ic_zi)

        # set_initial_conditions uses ds_i, us_i, us_j index arrays
        # but are those used outside set initial condition?
        # yes, yes they are.

    fig=None
    ax=None
    def prep_figure(self):
        plt.figure(1).clf()
        self.fig,self.ax=plt.subplots(num=1)
        self.ax.set_position([0,0,1,1])
        # self.grd.plot_edges(ax=self.ax,color='k',lw=0.5)
        # self.ax.axis('equal')
        self.ax.xaxis.set_visible(0)
        self.ax.yaxis.set_visible(0)

    def Xstep_output(self,n,ei,**kwargs):
        if self.fig is None:
            return
        if n%20:
            return

        self.prep_figure()
        ccoll=sim.grd.plot_cells(values=sim.hi,cmap='gray',ax=self.ax)
        cc=sim.grd.cells_center()
        ui = sim.get_center_vel(sim.uj)
        mags=utils.mag(ui)
        quiv=self.ax.quiver(cc[:,0],cc[:,1],ui[:,0],ui[:,1],mags,cmap=cm.rainbow,clim=[0,1.0])

        # plt.colorbar(ccoll)
        self.ax.quiverkey(quiv,0.1,0.1,0.5,"0.5 m/s")
        self.ax.axis('equal')
        self.fig.canvas.draw()
        plt.pause(0.001)


sim=SwampyWetDry(dt=10.0,ManN=0.025)
sim.get_fu=sim.get_fu_no_adv

sim.cg_tol=1e-10

h_deep=-5
h_shallow=5
W=200
L=1000

g=ugrid.UnstructuredGrid(max_sides=4)
g.add_rectilinear([0,0],[L,W],11,5)
g.orient_edges()

def depth_fn(xy):
    return h_deep + xy[:,1]/W * (h_shallow-h_deep)
g.add_node_field('node_depth',depth_fn(g.nodes['x']))
g.add_cell_field('cell_depth',depth_fn(g.cells_center()))

sim.set_grid(g)
sim.set_initial_conditions(h=-2)

inflow=np.array( [[0,0],
                  [0,W]] )

sim.add_bc( SWAMPpy.FlowBC(geom=inflow,Q=25) ) 

sim.prep_figure()
plt.draw()


@utils.add_to(sim)
def step_output(self,n,ei,**kwargs):
    if n%20: return
    
    ui=self.get_center_vel(self.uj)

    plt.figure(2).clf()
    fig,axs=plt.subplots(3,1,sharex=True,sharey=True,num=2)
    ax_h=axs[0]
    ax_ei=axs[1]
    ax_zi=axs[2]

    mag=utils.mag(ui)

    ccoll=g.plot_cells(values=sim.ei + sim.zi,ax=ax_h)
    plt.colorbar(ccoll,ax=ax_h,label='h=ei+zi')

    if 0:
        cc=g.cells_center()
        quiv=ax.quiver(cc[:,0],cc[:,1],ui[:,0],ui[:,1])

    ec=g.edges_center()
    quiv=ax_h.quiver( ec[:,0],ec[:,1], self.uj*sim.en[:,0],self.uj*sim.en[:,1],
                      color='b',scale=0.5)
    umag=0.1 # np.percentile(uj,90)
    ax_h.quiverkey(quiv,0.1,0.1,umag,"%.4g m/s"%umag)

    hj=self.hjstar
    g.plot_edges(color='k',lw=0.5,ax=ax_h,
                 labeler=lambda j,r: "%.2f"%hj[j])


    eicoll=g.plot_cells(values=self.ei,ax=ax_ei,
                        labeler=lambda i,r: "%.2f"%(self.ei[i]))
    plt.colorbar(eicoll,ax=ax_ei,label='ei')

    zicoll=g.plot_cells(values=-self.zi,ax=ax_zi,
                        labeler=lambda i,r: "%.2f"%(-self.zi[i]))
    plt.colorbar(zicoll,ax=ax_zi,label='-zi')

    axs[0].axis('equal')

    for ax in axs:
        plt.setp(ax.texts,clip_on=1)

    plt.draw()
    plt.pause(0.01)


(hi, uj, tvol, ei) = sim.run(tend=50000.)


##
# ei: eta
# sim.zi: bed elevation

# ei is allowed to go below -zi
# 
